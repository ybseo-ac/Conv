
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import lightning as L
import omegaconf

import dataloader
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from answer_generation.answer_tool import ready_dataset, generation_eos_cut, generation_len_cut
from finetune_tools import _ar_sampler_conditional, _sample_semi_ar
from r2ft_tools import _ddpm_topk_update, _ddpm_convolution_update
import diffusion
import sys
import json

overrides = sys.argv[1:]

GlobalHydra.instance().clear()
initialize(config_path="./configs/", version_base=None)
config = compose(config_name="config", overrides=overrides)

#### default ####
config.mode = 'sample_eval'
# config.model.length=1024

# config.parameterization = 'subs' 
###########################
omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)

original_cwd = os.getcwd()



# os.environ['NCCL_TIMEOUT_MS'] = '9000000'  #30분에서 2시간30분으로


rand_value = config.rand_value



#####################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(config.master_port)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

#################################


    
@torch.no_grad()
def generate0(model,x, timesteps, num_steps, dt , rank):
    p_x0_cache = None
    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device= rank) # shape 는 (batch size, 1)
        if model.sampler == 'ddpm':
            x = model._ddpm_update(x, t, dt)
        elif model.sampler == 'ddpm_topk':
            x = _ddpm_topk_update(model, x, t, dt)
        elif model.sampler == 'ddpm_convolution':
            x = _ddpm_convolution_update(model, x, t, dt)
        elif model.sampler == 'ddpm_cache':
            p_x0_cache, x_next = model._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
            if (not torch.allclose(x_next, x) or config.time_conditioning):
                # Disable caching
                p_x0_cache = None
            x = x_next
        else:
            x = model._analytic_update(x, t, dt)
            
            
        if config.sampling.eos_fill:  
            eos_found = ((x == model.tokenizer.eos_token_id).cumsum(dim=-1) >=2)
            x[eos_found] = model.tokenizer.eos_token_id        
    
    return x
    
            
def demo_basic(rank, world_size, test_data, generated_result, generated_results_dict): # run()
    print(f"Running basic DDP example on rank {rank}.")
    
    if rank==0:
        print(f"""
model : {config.init_from_checkpoint.init_file}
backbone : {config.backbone}
sampler: {config.sampling.predictor}
sampling steps: {config.sampling.steps}
sampling.semi_ar_bool steps: { config.sampling.semi_ar_bool}
generator : {config.generator}
batchsize : {config.batch_size}
parameterization : {config.parameterization}
output : answer_generation/generated/{config.category}/{config.generator}/
      """)

        if 'test_size' in config:
            print(f"restrict test size into {config.test_size}")
    
    
    setup(rank, world_size)
    torch.cuda.set_device(rank) ##


    tokenizer = dataloader.get_tokenizer(config)
    model = diffusion.Diffusion(config, tokenizer=tokenizer).to(rank)
    
    
        
    if config.init_from_checkpoint.init_file !='':
        state_dict = torch.load(config.init_from_checkpoint.init_file, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        missing_keys, unexpected_keys =model.load_state_dict(state_dict, strict=False)
        if rank==0:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

    data_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False) 
    dataloader_test = DataLoader(test_data, batch_size = config.batch_size,  sampler=data_sampler, drop_last=False)
    
    ddp_model = model
    ddp_model.eval()
    tqdm_disable = False if rank==0 else True

    #####
    eps=1e-5
    num_steps = model.config.sampling.steps
    timesteps = torch.linspace( 1, eps, num_steps + 1, device=rank)  # divide   1 -> 0  by num_step +1 
    dt = (1 - eps) / num_steps
    
    
    dataset=[]
    total_steps = len(dataloader_test) 
    for step, batch_data in tqdm(enumerate(dataloader_test) , total=total_steps, disable=tqdm_disable):
        xT = batch_data['xT'].to(rank)
        masked = xT == ddp_model.mask_index
        if config.sampling.semi_ar_bool:
            x, _ = _sample_semi_ar(ddp_model, batch_data)
        else:
            x = generate0(ddp_model, xT, timesteps, num_steps, dt, rank)
        
        for i in range(len(batch_data['id'])):
            dic ={}
            id = batch_data['id'][i]
            if type(id) ==torch.Tensor:
                id = id.tolist()
            dic['id'] = id
            dic['instruction'] = batch_data['instruction'][i]
            dic['output'] = tokenizer.decode(x[i][masked[i]])
            dic['dataset'] = batch_data['dataset'][i]
            dic['generator'] = config.generator
            if 'label' in batch_data:
                dic['label'] = batch_data['label'][i]

            dataset.append(dic)

    dist.barrier()
    torch.save(dataset, f"./tmp/{rand_value}_{rank}")
    cleanup()

"""
batch_data:
{'generator':['a','a','a'], 'id':[], 'instruction':[], 'xT':[]}
"""

def run_demo(demo_fn, world_size, test_data, generated_results, generated_results_dict):
    mp.spawn(demo_fn,  #
            args=(world_size, test_data, generated_results, generated_results_dict),
            nprocs=world_size,
            join=True)
    
 

if __name__ == "__main__":
    
    print(overrides)
    
    os.makedirs(f'answer_generation/generated/{config.category}/{config.generator}', exist_ok=True)
    
    
    """Main entry point for training."""
    L.seed_everything(config.seed)
    n_gpus = torch.cuda.device_count()

    world_size = n_gpus
    test_data = load_dataset('json', data_files=f'src_data/alpaca_eval/{config.eval_data}_questions.json')['train'] 
    
    if 'test_size' in config:
        test_data = test_data.select(range(config.test_size))


    tokenizer = dataloader.get_tokenizer(config)
    test_data = ready_dataset(config, test_data, tokenizer)


    with mp.Manager() as manager:
        generated_results = manager.list()
        generated_results_dict = manager.dict()

        run_demo(demo_basic, world_size, test_data, generated_results, generated_results_dict)

        generated_results = []
        for i in range(world_size):
            shard = torch.load(f"tmp/{rand_value}_{i}")
            generated_results.extend(shard)


            os.remove(f"tmp/{rand_value}_{i}")

        print(len(generated_results))
        print(generated_results[0]['output'][:500])

        dics = {}
        for line in generated_results:  # Removing redundunt data
            dics[line['id']] = line  
        sorted_values = [value for key, value in sorted(dics.items())]  # 


    with open(f'answer_generation/generated/{config.category}/{config.generator}/base.json', 'w') as f:
        json.dump(sorted_values, f, indent=4)



    sorted_values = generation_eos_cut(sorted_values, tokenizer)
    with open(f'answer_generation/generated/{config.category}/{config.generator}/eos_cut.json', 'w') as f:
        json.dump(sorted_values, f, indent=4)


    sorted_values = generation_len_cut(sorted_values, tokenizer, len0=500)
    with open(f'answer_generation/generated/{config.category}/{config.generator}/len500.json', 'w') as f:
        json.dump(sorted_values, f, indent=4)