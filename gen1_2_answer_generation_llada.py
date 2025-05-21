
import os
os.environ['NCCL_TIMEOUT_MS'] = '18000000'  #30분에서 5시간으로
os.environ["NCCL_BLOCKING_WAIT"] = "0"
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
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
import math
import torch.nn.functional as F
import numpy as np
from datetime import timedelta

overrides = sys.argv[1:]

GlobalHydra.instance().clear()
initialize(config_path="./configs/", version_base=None)
config = compose(config_name="config", overrides=overrides)

#### default ####
config.mode = 'sample_eval'
config.model.length=1024

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





rand_value = config.rand_value



#####################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12321'
    os.environ["NCCL_BLOCKING_WAIT"] = "0"  # deprecated
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=7200000))

def cleanup():
    dist.destroy_process_group()

#################################


def add_gumbel_noise(logits, temperature):
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@ torch.no_grad()
def generate(model, rank, prompt, steps=128, gen_length=1024, block_length=512, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=None):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(rank)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    num_blocks = math.ceil(gen_length / block_length)


    for num_block in range(num_blocks):
        steps0 = steps // num_blocks

        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps0)
        for i in range(steps0):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model.backbone.forward(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if model.config.backbone=='dit':
                    logits = model.forward(x, torch.tensor([0.],device=rank))
                else:
                    logits = model.backbone.forward(x).logits.to(torch.float32)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)  
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) 
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=rank)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf    

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=rank)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x
            
def demo_basic(rank, world_size, test_data, generated_result, generated_results_dict): # run()
    print(f"Running basic DDP example on rank {rank}.")
    
    if rank==0:
        print(f"""
model : {config.init_from_checkpoint.init_file}
sampling steps: {config.sampling.steps}
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
    batch_size = 1
    dataloader_test = DataLoader(test_data, batch_size = 1,  sampler=data_sampler, drop_last=False)
    
    ddp_model = model
    ddp_model.eval()
    tqdm_disable = False if rank==0 else True

    #####
    eps=1e-5
    num_steps = model.config.sampling.steps
    timesteps = torch.linspace( 1, eps, num_steps + 1, device=rank)  # 1 -> 0  을 num_step +1 으로 나눔.
    dt = (1 - eps) / num_steps
    
    tqdm_disable = False if rank==0 else True
    total_steps = len(dataloader_test)
    dataset=[]
    for i, data in tqdm(enumerate(dataloader_test), disable=tqdm_disable, total=total_steps):
        dic = {}
        input_ids = data['xT'].to(rank)
        masked = (input_ids == model.mask_index)
        prompt = input_ids[~ masked].unsqueeze(0)
        if config.model_max_length == 512:
            gen_length = config.model_max_length
        else:
            gen_length = config.model_max_length - prompt.size(-1)  
        out = generate(ddp_model, rank, prompt, steps=config.sampling.steps, gen_length=gen_length, block_length=config.sampling.semi_ar_stride_length, temperature=config.temperature, cfg_scale=0., remasking='low_confidence', mask_id=model.mask_index)
        id = data['id'][0]
        if type(id)==torch.Tensor:
            id = id.tolist()
        dic['id'] = id
        dic['instruction'] = data['instruction'][0]
        
        if len(out[0]) > len(masked[0]):
            masked = torch.concat((masked[0],  torch.ones(len(out[0]) - len(masked[0])).to(rank).to(masked.dtype))).unsqueeze(0)
        dic['output'] = tokenizer.decode(out[0][masked[0]])
        dic['dataset'] =data['dataset'][0]
        dic['generator'] = config.generator
        dataset.append(dic)

        dist.barrier()
    torch.save(dataset, f"./tmp/{rand_value}_{rank}")
    cleanup()

"""
batch_data:
{'generator':['a','a','a'], 'id':[], 'instruction':[], 'xT':[]}
"""

def run_demo(demo_fn, world_size, test_data, generated_results, generated_results_dict):
    mp.spawn(demo_fn,  # demo_fn  이  5번파일 run() 과 같음
            args=(world_size, test_data, generated_results, generated_results_dict),
            nprocs=world_size,
            join=True)
    
 

if __name__ == "__main__":
    
    print(overrides)
    
    os.makedirs(f'answer_generation/generated/{config.category}/{config.generator}', exist_ok=True)
    
    
    """Main entry point for training."""
    L.seed_everything(config.seed)
    n_gpus = torch.cuda.device_count() 
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    world_size = n_gpus
    test_data = load_dataset('json', data_files=f'src_data/alpaca_eval/{config.eval_data}_questions.json')['train'] # alpaca_eval_questions
    
    if 'test_size' in config:
        test_data = test_data.select(range(config.test_size))


    tokenizer = dataloader.get_tokenizer(config)
    tokenizer.model_max_length= config.model_max_length
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
        for line in generated_results:  #
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