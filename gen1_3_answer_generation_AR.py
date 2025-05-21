# yb_LAMA_learn_and_forget_test_multigpu.py 
# wloss2_e6_phi_여러 데이터 학습 및 eval.ipynb
# 거의 참고함

import os
os.environ['NCCL_TIMEOUT_MS'] = '18000000'  #30분에서 5시간으로
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
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
from finetune_tools import _ar_sampler_conditional
from repeat_dpo_tools import _ddpm_topk_update
import diffusion
import sys
import json
from datetime import timedelta
overrides = sys.argv[1:]
GlobalHydra.instance().clear()
initialize(config_path="./configs/", version_base=None)
config = compose(config_name="config", overrides=overrides)

#### default ####
config.mode = 'sample_eval'
config.model.length=1024

config.parameterization = 'ar' 
# config.backbone = 'ar' 
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
    os.environ['MASTER_PORT'] = '12356'
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    os.environ['NCCL_TIMEOUT_MS'] = '18000000'

    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=7200000))

def cleanup():
    dist.destroy_process_group()

#################################


    
# @torch.no_grad()
# def generate0(model,x, timesteps, num_steps, dt , rank):
#     p_x0_cache = None
#     for i in range(num_steps):
#         t = timesteps[i] * torch.ones(x.shape[0], 1, device= rank) # shape 는 (batch size, 1)
#         if config.sampler == 'ddpm':
#             x = model.module._ddpm_update(x, t, dt)
#         elif config.sampler == 'ddpm_topk':
#             x = _ddpm_topk_update(model.module, x, t, dt)
#         elif config.sampler == 'ddpm_cache':
#             p_x0_cache, x_next = model.module._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
#             if (not torch.allclose(x_next, x) or config.time_conditioning):
#                 # Disable caching
#                 p_x0_cache = None
#             x = x_next
#         else:
#             x = model.module._analytic_update(x, t, dt)
#     return x
    
            
def demo_basic(rank, world_size, test_data, generated_result, generated_results_dict): # run()
    print(f"Running basic DDP example on rank {rank}.")
    
    if rank==0:
        print(f"""
model : {config.init_from_checkpoint.init_file}
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
            
    # if rank==0:
        # print(model)
    data_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False) 
    dataloader_test = DataLoader(test_data, batch_size = config.batch_size,  sampler=data_sampler, drop_last=False)
    
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    ddp_model.eval()
    tqdm_disable = False if rank==0 else True

    #####
    # eps=1e-5
    # num_steps = model.config.sampling.steps
    # timesteps = torch.linspace( 1, eps, num_steps + 1, device=rank)  # 1 -> 0  을 num_step +1 으로 나눔.
    # dt = (1 - eps) / num_steps
    
    
    dataset=[]
    total_steps = len(dataloader_test) 
    for step, batch_data in tqdm(enumerate(dataloader_test) , total=total_steps, disable=tqdm_disable):
        with torch.no_grad():
            x, finetune_mask = _ar_sampler_conditional(model, batch_data)
            content_len = (batch_data['xT'] != model.mask_index).sum(dim=-1)
        
        
        # for i in range(batch_data['id'].size(0)):
        for i in range(len(batch_data['id'])):
            dic ={}
            id = batch_data['id'][i]
            if type(id) ==torch.Tensor:
                id = id.tolist()
            dic['id'] = id
            dic['instruction'] = batch_data['instruction'][i]
            dic['output'] = tokenizer.decode(x[i][finetune_mask[i].bool()][content_len[i]:])
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
    mp.spawn(demo_fn,  # demo_fn  이  5번파일 run() 과 같음
            args=(world_size, test_data, generated_results, generated_results_dict),
            nprocs=world_size,
            join=True)
    
 

if __name__ == "__main__":
    
    print(overrides)
    
    os.makedirs(f'answer_generation/generated/{config.category}/{config.generator}', exist_ok=True)
    
    
    """Main entry point for training."""
    L.seed_everything(config.seed)
    n_gpus = torch.cuda.device_count() # 타이탄 서버는 3개
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    world_size = n_gpus
    # test_data = load_dataset('json', data_files='src_data/alpaca_eval/alpaca_eval_questions.json')['train']
    test_data = load_dataset('json', data_files=f'src_data/alpaca_eval/{config.eval_data}.json')['train'] # alpaca_eval_questions
    
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
        for line in generated_results:  #중복제거
            dics[line['id']] = line  
        sorted_values = [value for key, value in sorted(dics.items())]  # 정렬

    with open(f'answer_generation/generated/{config.category}/{config.generator}/base.json', 'w') as f:
        json.dump(sorted_values, f, indent=4)



    sorted_values = generation_eos_cut(sorted_values, tokenizer)
    with open(f'answer_generation/generated/{config.category}/{config.generator}/eos_cut.json', 'w') as f:
        json.dump(sorted_values, f, indent=4)


    sorted_values = generation_len_cut(sorted_values, tokenizer, len0=500)
    with open(f'answer_generation/generated/{config.category}/{config.generator}/len500.json', 'w') as f:
        json.dump(sorted_values, f, indent=4)