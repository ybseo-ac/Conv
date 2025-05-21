import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # llada에서 워닝 없애려고 추가함

import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
import argparse
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import pickle 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from bitsandbytes.optim import AdamW
from torch.optim import SGD
import diffusion
from torch.utils.data.distributed import DistributedSampler
import lightning as L
import omegaconf

import dataloader
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from answer_generation.answer_tool import ready_dataset, generation_eos_cut, generation_len_cut
from finetune_tools import _ar_sampler_conditional, _sample_semi_ar, get_dataloaders_finetune
from repeat_dpo_tools import _ddpm_topk_update, _ddpm_convolution_update
import diffusion
import sys
import json
import numpy as np
import torch.nn.functional as F
import math
from collections import OrderedDict
from train_tools import _compute_loss, validation_step, on_validation_epoch_end, eval_step
from pytorch_lightning import Trainer

overrides = sys.argv[1:]
GlobalHydra.instance().clear()
initialize(config_path="./configs/", version_base=None)
config = compose(config_name="config", overrides=overrides)

#### default ####
config.mode = 'sample_eval'
config.model.length=1024
print(f"parameterization:  {config.parameterization}")
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



#####################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['NCCL_TIMEOUT_MS'] = '9000000'
    # os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

########## evaluation 관련





def demo_basic(rank, world_size, train_data, valid_data, trainer): # run()
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    torch.cuda.set_device(rank) ##

    if rank==0:
        print(overrides)
    model_name = config.data.tokenizer_name_or_path
    
    
    
    model = diffusion.Diffusion(
    config, tokenizer=valid_data.tokenizer).to(rank)
    
    if config.init_from_checkpoint.bool:
      state_dict = torch.load(config.init_from_checkpoint.init_file)
      missing_keys, unexpected_keys = model.load_state_dict(state_dict['state_dict'] ,strict=False)
      if rank==0:
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    
    
    model.trainer = trainer
    

    

    
    #####################

    train_sampler = DistributedSampler(train_data.dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
    dataloader_train = DataLoader(train_data.dataset,batch_size=config.loader.batch_size,
      pin_memory=config.loader.pin_memory,
      sampler=train_sampler)
    
    valid_sampler = DistributedSampler(valid_data.dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=config.seed) 
    dataloader_valid= DataLoader(valid_data.dataset,
                                 batch_size=config.loader.eval_batch_size,
      pin_memory=config.loader.pin_memory,
      generator=None,
      sampler=valid_sampler)

    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    for name, param in model.backbone.named_parameters():
      if 'lm_head' in name or '.norm' in name:
          param.data = param.data.to(dtype=torch.float32)
        
    param_list = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
          param.requires_grad_()
        if param.requires_grad:
            param_list.append(param)

    optimizer = torch.optim.AdamW(param_list,
      lr=config.optim.lr,
      betas=(config.optim.beta1,
             config.optim.beta2),
      eps=config.optim.eps,
      weight_decay=config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      config.lr_scheduler, optimizer=optimizer)

    tqdm_disable = False if rank==0 else True
    
    global_step=0
    epoch=0
    for epoch in range(0, 5):#args.max_epochs):  
        ddp_model.module.backbone.train()
        
        if epoch >-1:

            optimizer.zero_grad()
            ddp_model.module.backbone.train()
            ddp_model.module.noise.train()
            
            total_steps = len(dataloader_train)
            loss_list = []
            for step, batch_data in tqdm(enumerate(dataloader_train), disable=tqdm_disable, total=total_steps):
                batch_data = {k: v.to(rank, non_blocking=True) for k, v in batch_data.items()}
                loss, metrics = _compute_loss(ddp_model.module, batch_data, 'train')
                # loss = ddp_model.module._compute_loss( batch_data, 'train')

                if torch.isnan(loss).any():
                    print("nan 있음")
                loss = loss / config.trainer.accumulate_grad_batches # default 1

                loss.backward()
                loss_list.append(loss.clone().detach())

                # gradiant accumulation # accum_step=1  이면 accum 안함
                if ((step + 1) % config.trainer.accumulate_grad_batches ==0) or (step + 1) == len(dataloader_train) :
                    # if rank==0:
                        # print(f"{step+1}  {len(dataloader_train)}")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    metric_values = metrics.compute()
                    dist.barrier()
                    global_step +=1
                    if rank==0:
                      
                      metric_values = metrics.compute()  # Compute all stored metrics
                      logged_metrics = {f"{key}": value.item() for key, value in metric_values.items()}
                      current_lr = optimizer.param_groups[0]['lr']
                      logged_metrics['global_step'] = global_step
                      logged_metrics['learning_rate'] = current_lr
                      logged_metrics['epoch'] = epoch
                      with torch.no_grad():
                        loss_mean = torch.tensor(loss_list).mean().item()
                      logged_metrics['train/loss'] = loss_mean
                      loss_list=[]
                      
                      ddp_model.module.trainer.logger.log_metrics(logged_metrics)
                      ddp_model.module.train_metrics.reset()
                      
                    dist.barrier()

                    if global_step % config.trainer.val_check_interval == 0:
                        with torch.no_grad():
                          eval_step(config, ddp_model,rank, world_size, dataloader_valid, global_step, epoch, )
                        
                        dist.barrier()
            if rank==0:
                print(f"{epoch} epoch / Loss : {loss.detach().item()}") ## loss.item() : loss 값
            with torch.no_grad():
              eval_step(config, ddp_model,rank, world_size, dataloader_valid, global_step, epoch, )
                        

            dist.barrier()

    
        # ######eval#####

            
        with torch.no_grad():
        
                  
          if rank==0:
              # self.model
              state_dict = OrderedDict()
              for name, param in ddp_model.module.named_parameters():
                  if param.requires_grad:
                      state_dict[name] = param.clone().detach().cpu()

              ## save
              result_dic = {'config': {** config},
              }
              result_dic['config']['world_size'] = world_size
              result_dic['state_dict']=state_dict

              torch.save(result_dic, f'outputs/{config.wandb.name}/cp_ep{epoch}_gbstep{global_step}.ckpt')

           
                
        # dist.barrier()
        
    cleanup()


def run_demo(demo_fn, world_size, train_data, valid_data,  trainer):
    mp.spawn(demo_fn,  
            args=(world_size, train_data, valid_data, trainer),
            nprocs=world_size,
            join=True)
    
 

if __name__ == "__main__":
    L.seed_everything(config.seed)
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    os.makedirs(f'outputs/{config.wandb.name}', exist_ok=True )
    
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
    trainer = Trainer(logger=wandb_logger, default_root_dir=os.getcwd())
    # trainer=None
    
    
    # model_name = "GSAI-ML/LLaDA-8B-Instruct"
    model_name = config.data.tokenizer_name_or_path
    tokenizer = dataloader.get_tokenizer(config)
    train_ds, valid_ds = get_dataloaders_finetune(config, tokenizer) #


    with mp.Manager() as manager:
        # learned_result = manager.list()
        learned_result = manager.dict()
        no_forget_result = manager.dict()
        loss_result = manager.dict()

        unchanged_ppl = manager.list()
        # eval_result= manager.dict()

        run_demo(demo_basic, world_size, train_ds, valid_ds,  trainer)

