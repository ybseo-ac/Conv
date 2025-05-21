# config 불러오기
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import dataloader
import diffusion
import torch
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from datasets import load_dataset, Dataset, DatasetDict
import sys

overrides = sys.argv[1:]
GlobalHydra.instance().clear()
initialize(config_path="./configs/", version_base=None)
config = compose(config_name="config", overrides=overrides)

config.mode = 'sample_eval'
if config.model_type=='small':
    config.eval.checkpoint_path='kuleshov-group/mdlm-owt'
    config.data.tokenizer_name_or_path ='kuleshov-group/mdlm-owt'
    config.backbone='hf_dit'
elif config.model_type=='large':
    config.eval.checkpoint_path='GSAI-ML/LLaDA-8B-Instruct'
    config.data.tokenizer_name_or_path ='GSAI-ML/LLaDA-8B-Instruct'
    config.backbone='llada'
elif config.model_type=='llama':
    config.eval.checkpoint_path='meta-llama/Meta-Llama-3-8B'
    config.data.tokenizer_name_or_path ='meta-llama/Meta-Llama-3-8B'
    config.backbone='llama'
config.model.length=1024
config.sampling.predictor='ddpm'
config.parameterization = 'subs'  # 이거는 여기서 하지 않으면 모델이 이렇게 안만들어짐

def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')

tokenizer = dataloader.get_tokenizer(config)

if config.model_type=='small':
    tokenizer.mask_token_id = tokenizer.encode("<|mdm_mask|>")[0]
elif config.model_type=='large':
    tokenizer.model_max_length =1024
    
model = diffusion.Diffusion( config, tokenizer=tokenizer)

## load dataset
ds = load_dataset('tatsu-lab/alpaca')


from tqdm import tqdm
processed_dataset = []
for i, data in tqdm(enumerate(ds['train'])):
    if data['output']=='':
        pass
    else:
        input0=''
        if data['input'] =="":
            input0=''
        else:
            input0=f'{data["input"]}\n\n'
        
        response = data['output']
        
        ### xt (fully noised)
        text_before_response = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{data['instruction']}\n\n{input0}### Response:\n"""
        token_before_response = tokenizer.encode(text_before_response, add_special_tokens=True, return_tensors='pt')[0,:-1] # remove latter EOS
        quary_len = len(token_before_response)
        mask_len = config.model.length - len(token_before_response)  # attach EOS to the front and the end
        xt = torch.concat((token_before_response ,  torch.ones(mask_len) * model.mask_index), dim=-1).int()  # shape : [1024]
        
        #x0 (original)
        text_after_response = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{data['instruction']}\n\n{input0}### Response:\n{response}"""
        output = tokenizer(text_after_response, add_special_tokens=True, return_tensors='pt')
        
        token_after_response = output.input_ids.squeeze()[:model.config.model.length]
        attention_mask_after_response = output.attention_mask.squeeze()[:model.config.model.length]
        eos_len = model.config.model.length - len(token_after_response)
        x0 = torch.concat((token_after_response ,  torch.ones(eos_len) * tokenizer.eos_token_id), dim=-1).int()
        
        x0[-1] = tokenizer.eos_token_id  # in case len(text) > 1024 

        attention_mask = torch.concat((attention_mask_after_response, torch.zeros(eos_len)), dim=-1).int()  # 
        attention_mask[:quary_len] = 0  
        
        processed_dataset.append({'input_ids': x0, 'x0':x0, 'xT':xt, 'attention_mask':attention_mask})
    

hf_dataset = Dataset.from_list(processed_dataset)



if config.model_type=='small':
    hf_dataset.save_to_disk("src_data/ft_data/tokenized_alpaca_instruction") 
elif config.model_type=='large':
    hf_dataset.save_to_disk("src_data/ft_data/tokenized_alpaca_instruction_large") 
elif config.model_type=='llama':
    hf_dataset.save_to_disk("src_data/ft_data/tokenized_alpaca_instruction_llama") 
