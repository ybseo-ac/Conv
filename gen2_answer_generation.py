# do4_1_answer_generation.py


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

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from answer_generation.answer_tool import ready_dataset, generation_eos_cut, generation_len_cut
import sys
import json
from generate_yb import generate, generate_yb, generate_llada_convolution, generate_semiar, generate_slide_annealing_update
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from types import SimpleNamespace

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


### small model 위한것
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model  # now properly registered

    @property
    def device(self):
        try:
            return next(self.backbone.parameters()).device
        except (StopIteration, AttributeError):
            return None
        
    def forward(self, *args, **kwargs):
        # Inject sigma=None by default unless already passed
        if 'sigma' not in kwargs:
            kwargs['sigma'] = torch.tensor([0.], device=self.device)
        a, b = self.backbone(*args, **kwargs)
        result = SimpleNamespace()
        result.logits = a
        return result

    def __getattr__(self, name):
        if name == "backbone":
            return super().__getattr__(name)
        return getattr(self.backbone, name)

    def __setattr__(self, name, value):
        if name == "backbone":
            super().__setattr__(name, value)
        else:
            setattr(self.backbone, name, value)

    
#####################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(config.master_port)

    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

#################################


            
def demo_basic(rank, world_size, test_data, generated_result, generated_results_dict): # run()
    print(f"Running basic DDP example on rank {rank}.")
    
    if rank==0:
        print(f"""
model : {config.model_path}
model : {config.init_from_checkpoint.init_file}
sampler: {config.decode_type}
sampling steps: {config.steps}
generator : {config.generator}
batchsize : {config.batch_size}
parameterization : {config.parameterization}
output : answer_generation/generated/{config.category}/{config.generator}/
      """)

        if 'test_size' in config:
            print(f"restrict test size into {config.test_size}")
    
    
    setup(rank, world_size)
    torch.cuda.set_device(rank) ##

    model_kwargs = {}
    
    if 'mdlm' in config.model_path:
        model = AutoModelForMaskedLM.from_pretrained(config.model_path, trust_remote_code=True, cache_dir="your_cache" , **model_kwargs).to(rank)
        model = ModelWrapper(model.backbone)
        tokenizer = AutoTokenizer.from_pretrained('gpt2',  cache_dir="your_cache" )
    else:
        model= AutoModel.from_pretrained(config.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir="your_cache" , **model_kwargs).to(rank)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path,  cache_dir="your_cache" )
    model.eval()
    
    if config.lora.bool :
        print("!!!!!!  Lora initialilze !!!!!!!!")
        from peft import LoraConfig, get_peft_model
        import bitsandbytes as bnb
        peft_config = LoraConfig(
            lora_alpha = config.lora.lora_alpha,
            lora_dropout= config.lora.lora_dropout,
            r=config.lora.lora_r,
            bias="none",
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                # 'ff_proj',
                # 'up_proj'
            ],
        )
        model = get_peft_model(model, peft_config)
        
    if config.init_from_checkpoint.init_file !='':
        state_dict = torch.load(config.init_from_checkpoint.init_file, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        if 'mdlm' not in config.model_path:
            state_dict = {
            k[len("backbone."): ] if k.startswith("backbone.") else k: v for k, v in state_dict.items() }
            
        missing_keys, unexpected_keys =model.load_state_dict(state_dict, strict=False)
        if rank==0:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

    data_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False) 
    dataloader_test = DataLoader(test_data, batch_size = config.batch_size,  sampler=data_sampler, drop_last=False)
    
    tqdm_disable = False if rank==0 else True

    #####
    steps = config.sampling.steps
    
    
    #####
    
    dataset=[]
    total_steps = len(dataloader_test) 
    for step, batch_data in tqdm(enumerate(dataloader_test) , total=total_steps, disable=tqdm_disable):
        prompt = batch_data['input_ids'].to(rank)
        
        
        if config.decode_type=='llada':
            generated_answer = generate(model, prompt, steps=steps, gen_length=config.gen_length, block_length=config.block_length, temperature=config.temperature, cfg_scale=config.cfg, remasking=config.remasking, mask_id=config.mask_id) 
            
        elif config.decode_type in ['conv','topk','categorical', 'topk_penalty', 'remdm_loop_topk', 'remdm_loop_conv']:
            generated_answer = generate_yb(model, tokenizer, prompt, steps=config.steps, gen_length=config.gen_length, block_length=config.block_length, temperature=config.temperature, mask_id=config.mask_id, eos_fill= config.eos_fill, decode_type=config.decode_type, k=config.sampling.topk_k, repetition_penalty = config.sampling.repetition_penalty, 
            alpha_on=config.sampling.alpha_on, t_on=config.sampling.t_on, t_off=config.sampling.t_off, eta=config.sampling.eta, bidirection=config.bidirection, bi_what = config.bi_what)  
            # block_length = kernel size
            # bidirection : 멘 마지막에 삽입
            # bid_what  :  뭘 삽입할지.  eos, '', None 들어가면 걍 eos token.
        
        elif config.decode_type in ['slide']:
            generated_answer = generate_slide_annealing_update(model, tokenizer, prompt, steps=config.steps, gen_length=config.gen_length, block_length=config.block_length, temperature=config.temperature, mask_id=config.mask_id, eos_fill= config.eos_fill, decode_type=config.decode_type, k=config.sampling.topk_k, repetition_penalty = config.sampling.repetition_penalty, 
            alpha_on=config.sampling.alpha_on, t_on=config.sampling.t_on, t_off=config.sampling.t_off, eta=config.sampling.eta, bidirection=config.bidirection, bi_what = config.bi_what) 
            
        elif config.decode_type == 'lladaconv':
            generated_answer = generate_llada_convolution(model, prompt, steps=config.steps, gen_length=config.gen_length, block_length=config.block_length, temperature=config.temperature, cfg_scale=config.cfg, remasking=config.remasking, mask_id=config.mask_id)
            
        elif config.decode_type == 'semiar':
            generated_answer = generate_semiar(model, tokenizer, prompt, steps=config.steps, gen_length=config.gen_length, block_length = config.block_length, temperature=config.temperature, mask_id=config.mask_id, k=config.sampling.topk_k, conv_mult=1, eos_fill=False, decode_type='topk')
        else:
            raise Exception("wrong!!!!!!!!!!!!!")
        
        # for i in range(batch_data['id'].size(0)):
        for i in range(len(batch_data['id'])):
            dic ={}
            id = batch_data['id'][i]
            if type(id) ==torch.Tensor:
                id = id.tolist()
            dic['id'] = id
            dic['instruction'] = batch_data['instruction'][i]
            dic['output'] = tokenizer.decode(generated_answer[i,prompt.shape[-1]:])
            dic['dataset'] = batch_data['dataset'][i]
            dic['generator'] = config.generator
            if 'label' in batch_data:
                dic['label'] = batch_data['label'][i]

            dataset.append(dic)

    # dist.barrier()
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
    
    if os.path.isdir(f'answer_generation/generated/{config.category}/{config.generator}'):
        print(f"{config.category}/{config.generator}  이미 있음############")
        exit()
    
    os.makedirs(f'answer_generation/generated/{config.category}/{config.generator}', exist_ok=True)
    
    
    """Main entry point for training."""
    L.seed_everything(config.seed)
    n_gpus = torch.cuda.device_count() # 타이탄 서버는 3개
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    world_size = n_gpus
    test_data = load_dataset('json', data_files=f'src_data/alpaca_eval/{config.eval_data}_questions.json')['train'] # alpaca_eval_questions
    
    if 'test_size' in config:
        test_data = test_data.select(range(config.test_size))

    if 'mdlm' in config.model_path:
        tokenizer = AutoTokenizer.from_pretrained('gpt2',  cache_dir="your_cache" )
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path,  cache_dir="your_cache" )
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

