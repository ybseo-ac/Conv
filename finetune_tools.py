import torch
import utils
from dataclasses import dataclass
from repeat_dpo_tools import _ddpm_topk_update, _ddpm_topk_caching_update
from datasets import load_from_disk
from tqdm import tqdm
@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor
  loss_w: torch.FloatTensor = None
  loss_l: torch.FloatTensor = None


def q_xt_finetune(self, x, xT, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
        data with   system + instruction + response + eos paddings
      xT:  totally unmasked.   system + instruction + response + mask paddings
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    if self.config.class_noise_type== 'absorbing':
        dice = torch.rand( * x.shape, device=x.device)
        query_part =  (xT != self.mask_index)
        dice[query_part] = 1  # never maksed
        move_indices = dice < move_chance
        xt_cl = torch.where(move_indices, self.mask_index, x)
        return xt_cl
    
    else:
        raise Exception("!!!!! Finetune forward pass not defined !!!!!!!!!!")
    


def _forward_pass_diffusion_finetune(self, x0, xT, attention_mask):
    t = self._sample_t(x0.shape[0], x0.device) # 데이터가 4개면 4개의 0~1 수 뽑음 (랜덤 time)
    if self.T > 0: # T=0이면 t:0~1 / T>0 이면 t:0~T (근데 다시 0~1로 discrete하게 쪼갬)
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables: # False
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None]) # 풀어보면 그냥 t 그자체임 ***    # sigma = -log(1-t)
      
      
    xt = q_xt_finetune(self, x0, xT, move_chance)  # xT != xt.  
    model_output = self.forward(xt, unet_conditioning) # logprobability (batch X sentence length X vocab size)  # unet_conditioning 사실상 쓰이지 않음
    utils.print_nans(model_output, 'model_output')
    
    
    # 아래는 pretrain과 같음.
    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs': # discrete time SUBS 인듯
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    loss = - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]
    
    return loss




  
def _loss_finetune(self, x0, xT, attention_mask, prefix=None):
    if self.parameterization == 'ar':
        return ar_sft_loss(self, x0, attention_mask)
      
    else:
        loss = _forward_pass_diffusion_finetune(self, x0, xT, attention_mask)
        
        if self.config.finetune.attention_cover=='response':
            nlls = loss * attention_mask
            count = attention_mask.sum()
        elif self.config.finetune.attention_cover=='response_and_eos':
            masked  = (xT ==self.mask_index).int()
            nlls = loss * masked
            count = masked.sum()
            #### count 어떻게#####
        
        batch_nll = nlls.sum()
        token_nll = batch_nll / count
        
    
        return Loss(loss=token_nll,
                  nlls=nlls,
                  token_mask=attention_mask)
        
##### sample ###

@torch.no_grad()
def _sample_conditional(self, batch, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar': # 여기로는 안오긴 함.**
      return _ar_sampler_conditional(self, batch)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
      
    x = batch['xT'].to(self.device)
    finetune_mask = (x == self.mask_index).int()  # 직접 생성한거   를 나타내게 될 것
    
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_topk':
        x = _ddpm_topk_update(self, x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      elif self.sampler == 'ddpm_topk_cache':
          p_x0_cache, x_next = _ddpm_topk_caching_update(self, x, t, dt, p_x0=p_x0_cache)
          if (not torch.allclose(x_next, x) 
              or self.time_conditioning):
              # Disable caching
              p_x0_cache = None
          x = x_next
      else:
        x = self._analytic_update(x, t, dt)
      
      if self.config.sampling.eos_fill:  
        eos_found = ((x == self.tokenizer.eos_token_id).cumsum(dim=-1) >=2)
        x[eos_found] = self.tokenizer.eos_token_id
        
    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning)
        if type(x)==tuple:
          x = x[0].argmax(dim=-1)
        else:
          x = x.argmax(dim=-1)
    
    if self.config.val_eos_off: # 이 변수는 여기서만 작용함.
        finetune_mask = finetune_mask * (x != self.tokenizer.eos_token_id).int()  # 생성한 것 중 eos 가 아닌 부분   => 직접 생성한거고 eos가 아닌 부분.   
    return x, finetune_mask



# conditional은 eos 토큰이 많아서 이를 제외하도록 함.
# prior = -inf 인 토큰을 처리하도록 함.
def prior_std_mean_calcul_conditional(model, samples, finetune_mask): 
    #finetune_mask : question 을 제외한 부분 =1    //  val_eos_off==True 라면  eos 부분 =0 (디폴트 Fault) 
    # [batch x  len ]
    
    token_prior = model.backbone.token_prior_for_eval
    min_prior = token_prior[token_prior != -torch.inf].min()  # -inf 가 아닌 것중 가장 낮은 prior
    not_eos_bool = samples != model.tokenizer.eos_token_id
    not_eos_bool = not_eos_bool * (finetune_mask.bool())
    data_prior = model.backbone.token_prior_for_eval[samples]
    mean_list =[]
    std_list =[]
    for i, data0 in enumerate(data_prior):
      data_wo_eos = data0[not_eos_bool[i]] # eos인 부분, mask인 부분 날려줌.
      # std
      std_list.append((data_wo_eos * 1000).std())
      # mean
      data_wo_eos = data_wo_eos.log()
      # data_wo_eos[data_wo_eos == -torch.inf] = min_prior.log()
      data_wo_eos[data_wo_eos != -torch.inf] = min_prior.log()
      mean_list.append(data_wo_eos.mean())

    prior_std = torch.stack(std_list)  # prior 의 std .  [batchsize]  #  eos 제외한 토큰으로 std 계산
    prior_mean = torch.stack(mean_list) # log prior 의 mean
    prior_mean_MSE = (prior_mean - model.backbone.token_prior_logs)**2
    prior_std_MSE = (prior_std - model.backbone.token_prior_stds)**2

    return prior_mean, prior_mean_MSE , prior_std, prior_std_MSE
    # model.backbone.token_prior_logs  : all_data_logs 의 median
    # model.backbone.token_prior_stds  : all_data_logs 의 median
    
    

#################### get dataloader ###################
def get_dataloaders_finetune(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
    num_gpus = torch.cuda.device_count()
    assert (config.loader.global_batch_size
            == (config.loader.batch_size
                * config.trainer.num_nodes
                * num_gpus
                * config.trainer.accumulate_grad_batches))
    assert (config.loader.global_batch_size
            == (config.loader.batch_size
                * config.trainer.num_nodes
                * num_gpus
                * config.trainer.accumulate_grad_batches))
    if config.loader.global_batch_size % (
        num_gpus * config.trainer.accumulate_grad_batches) != 0:
        raise ValueError(
        f'Train Batch Size {config.training.batch_size}'
        f'not divisible by {num_gpus} gpus with accumulation '
        f'{config.trainer.accumulate_grad_batches}.')
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
        f'Eval Batch Size for {config.eval.batch_size} '
        f'not divisible by {num_gpus}.')

    dataset0 = load_from_disk(config.finetune.dataset)
    dataset0.set_format(type="torch")
    split_ds = dataset0.train_test_split(test_size=config.finetune.valid_size, seed=42)
    train_set = split_ds['train']
    valid_set = split_ds['test']  
    
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=True)
    train_loader.tokenizer = tokenizer   

    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=False,
      generator=None)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer
    return train_loader, valid_loader
    
###################################  finetune  rDPO  #######################

# noise type:  query 이후는 모두 padding. query 내에서 무작위로 phase 추출. 해당 phase를 반복함
def q_xt_finetune_repeat_naive(self, x, xT, move_chance):
    unmasked = (xT != self.mask_index)
    answer_query_len = (x != self.tokenizer.eos_token_id).sum(dim=-1) +1 # 멘 처음 eos 포함해서 +1
    eos_len = (x == self.tokenizer.eos_token_id).sum(dim=-1) -1  # qa 이후 eos. 첫번째는 제외.
    repeat_mask = torch.zeros_like(xT) # repeat response 가 된 부분을 표시하기 위함. def repeat_q_xt 에서  'mask' 의 역할. 근데 mask할 필요는 없고, 길이가 유동적이므로 이렇게 씀.
    
    xb = xT.clone()
    for i in range(x.size(0)):
        query_index = unmasked[i].sum()
        a = torch.randperm(query_index)[-2:] +1  # 멘앞 eos는 피하고, 멘 뒤 \n 까지 포함시킴
        a= a[a.argsort()]
        # rand_val.append(a)
        repeat = x[i, a[0]:a[1]]
        rand_permute = torch.randint(0, len(repeat),(1,))
        rand_add = torch.min(eos_len[i] , torch.tensor(30))  # 원래 답변 길이보다 최대 30 까지 repeat 구간을 늘림.  eos_len 이 얼마 안남았을 수 있어서 min으로 설정
        
        rand_add = torch.randint(rand_add, (1,), device=x.device)
        rand_add = torch.max(rand_add, torch.tensor(1))
        # max(rand_add, 1) 이 들어간 이유: repeat mask 는  query_index: answer_query_len[i]  여기서 우로 1칸 더 포함시켜야만 attention_mask와 길이가 같아짐. 왜냐면 attention_mask 는 본문 외에 마지막 eos_token 을 하나 더 가지고 있기 때문임.
        # 길이가 안맞으면 grad explod이 생김 (왜인지는 모름)
        for mask_i in range(query_index, answer_query_len[i] + rand_add):
            xb[i][mask_i] = repeat[(mask_i + rand_permute) % len(repeat) ]
        
        repeat_mask[i, query_index: answer_query_len[i] + rand_add] =1
    repeat_mask = repeat_mask.bool()
    xt = xT.clone()
    
    return xb, xt, repeat_mask   # xt = xT

# noise type:  query 이후 ~ answer 전까지의 지점(c) 이후부터 padding.  c-g ~ c 사이의 repeat phase를  c ~ c+z  까지 반복.
def q_xt_finetune_repeat(self, x, xT, move_chance):
  b= (x != self.tokenizer.eos_token_id ).sum(dim=-1) +1  # x0 의 경계   [batch_size]
  a =  (xT != self.tokenizer.pad_token_id).sum(dim=-1)  # xT 의 경계  [batch_size]
  
  # print(f"{b}    {a}   {self.tokenizer.eos_token_id}   {self.tokenizer.pad_token_id}")
  repeat_mask = torch.zeros_like(xT, device=x.device)
  xt = x.clone()
  xb = xt.clone()
  range_c = b - a  # [batch_size]
  repeat_longer_than_x0 = self.config.finetune.repeat_longer_than_x0

  for i, max_c in enumerate(range_c):
      c = torch.randint(max_c, (1,), device=x.device) + a[i]  # xt 의 경계 (= repeat phase 의 우측 경계)
      g = torch.randint(1,15, (1,), device=x.device)  # length of repeated phase
      # z = torch.randint(30, 80 , (1,), device=x.device)  # length of area to repeat over
      z = torch.randint(30, self.config.finetune.max_z , (1,), device=x.device)  # length of area to repeat over
      xt[i, c:] = self.tokenizer.pad_token_id  # 
      repeat = xt[i, c - g : c]
      xb[i] = xt[i]
      rand_permute = torch.randint(0, len(repeat),(1,), device=x.device)
      end0 = torch.min(torch.tensor(x.size(-1)), c + z)
    
      if repeat_longer_than_x0:
        end0 = torch.max(end0, b[i])
      else:
        end0 = c + z

      for mask_i in range(c, end0):
          xb[i][mask_i] = repeat[(mask_i + rand_permute) % len(repeat) ]
          repeat_mask[i, mask_i] = 1

      if torch.rand(1) <= self.config.finetune.eos_p:
        eos0= torch.randint(c, end0 -1, (1,), device=x.device)  # 마지막칸에 eos 넣는 행위는 unlearning 하지 않기위해 -1
        xb[i, eos0] = self.tokenizer.eos_token_id   
      repeat_mask[i, c: end0] = 1   
      
        
  return xb, xt, repeat_mask 


def _forward_pass_diffusion_finetune_rDPO(self, x0, xT, attention_mask): 
    t = self._sample_t(x0.shape[0], x0.device) 
    if self.T > 0: # T=0이면 t:0~1 / T>0 이면 t:0~T (근데 다시 0~1로 discrete하게 쪼갬)
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables: # False
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None]) # 풀어보면 그냥 t 그자체임 ***    # sigma = -log(1-t)
    ##### move_chance, t   사실상 쓰이지 않음 #####



    xb, xt, repeat_mask = q_xt_finetune_repeat(self, x0, xT, move_chance) # repeat_mask : repeat_q_xt 에서의 mask 역할
    # xb : rejected label   with repeated patterns
      
    # 아래부터 loss function 구간   

    model_output = self.forward(xt, unet_conditioning) # logprobability (batch X sentence length X vocab size)  # unet_conditioning 사실상 쓰이지 않음

    utils.print_nans(model_output, 'model_output')

    if self.parameterization in ['d3pm', 'subs']:
        
        # yw > yl
        log_p_w = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)  * attention_mask
        log_p_l = torch.gather(
            input=model_output,
            dim=-1,
            index=xb[:, :, None]).squeeze(-1) * repeat_mask
        
        denom_w = attention_mask.sum(dim=-1)
        denom_l = repeat_mask.sum(dim=-1)
        
        # log_p_l[:,0] = 0  # 여기선 이거 필요 없을듯
        # log_p_l[:,-1] = 0
        
        loss = - torch.log(torch.sigmoid( ((self.config.repeat_dpo.beta_w * log_p_w.sum(dim=-1) /denom_w) - (self.config.repeat_dpo.beta_l * log_p_l.sum(dim=-1) /denom_l))- self.config.repeat_dpo.gamma)) - ((self.config.repeat_dpo.beta_a * log_p_w.sum(dim=-1))/denom_w)
        
        loss = loss.sum()
        
        with torch.no_grad():
            history_log_p_w = -torch.log(torch.sigmoid((log_p_w.sum(dim=-1)/denom_w).sum()))
            history_log_p_l = -torch.log(torch.sigmoid((log_p_l.sum(dim=-1)/denom_l).sum()))
        
        return loss, history_log_p_w, history_log_p_l
    else:
        raise Exception("!!!!! parameterization not yet prepared !!!!!!!")



def _loss_finetune_dpo(self, x0, xT, attention_mask, prefix=None):
    # (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(x0, attention_mask)

    if self.parameterization=='ar':
      loss, history_log_p_w, history_log_p_l  = _forward_pass_AR_finetune_rDPO(self,  x0, xT, attention_mask)
    else:
      loss, history_log_p_w, history_log_p_l  = _forward_pass_diffusion_finetune_rDPO(self,  x0, xT, attention_mask)
    
    # if prefix=='train':
    #     self.log(name='trainer/loss_w', value=history_log_p_w.item(), on_step=True, on_epoch=False, sync_dist=True)
    #     self.log(name='trainer/loss_l', value=history_log_p_l.item(), on_step=True, on_epoch=False, sync_dist=True)
    
    # if prefix=='val':
    #     self.valid_dpo_w_metrics.update(torch.ones_like(attention_mask, device=attention_mask.device)* history_log_p_w.detach())
    #     metrics_w = self.valid_dpo_w_metrics
    #     self.valid_dpo_l_metrics.update(torch.ones_like(attention_mask, device=attention_mask.device)* history_log_p_l.detach())
    #     metrics_l = self.valid_dpo_l_metrics
    #     self.log_dict(metrics_w ,on_step=False,on_epoch=True,sync_dist=True)
    #     self.log_dict(metrics_l ,on_step=False,on_epoch=True,sync_dist=True)
        
    
    return Loss(loss=loss, token_mask=attention_mask, nlls=torch.ones_like(attention_mask, device=attention_mask.device)* loss.detach(), loss_w=history_log_p_w, loss_l=history_log_p_l )



#########################  AR  #################################

## sft loss
def ar_preprocess(self, x0, attention_mask):
  ##. diffusion LM SFT 데이터셋은 좌로 정렬, 패딩은 우측임.  이를 우측 정렬로 바꿔주는 장치. (expp10_2_4.ipynb)
  # print(f"{self.tokenizer.eos_token_id}  {self.tokenizer.pad_token_id}")
  assert x0.size(-1) == self.tokenizer.model_max_length
  content_len = (x0 != self.tokenizer.eos_token_id).sum(dim=-1)
  num_cols = attention_mask.shape[1]
  batch_indices = torch.arange(num_cols).repeat(x0.shape[0], 1).to(x0.device)  # [[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]]
  x0[batch_indices > content_len.unsqueeze(-1) +1] = self.tokenizer.pad_token_id  # eos ~~~ eos pad pad pad 
  # 문장을 eos로 감싸고 나머지는 pad로 바꿔줌
  if self.config.ar.eos_after_gen:
      min_mask_len = torch.min((x0 == self.mask_index).sum(dim=-1))
  else:
      min_mask_len = 0
  shifted_indices = (batch_indices - self.tokenizer.model_max_length + content_len.view(-1, 1) +2 + min_mask_len)  % num_cols   #  +2 : eos 까지 같이 옮기기 위함
  new_input_ids = torch.gather(x0, 1, shifted_indices).to(x0.device)
  new_attention_mask = torch.gather(attention_mask, 1, shifted_indices)

  if self.config.ar.eos_after_gen:  
      new_input_ids[:,content_len.max() +1 :] = self.tokenizer.eos_token_id
      new_attention_mask[:,content_len.max() +1 :] = 1
  # 원래 _maybe_sub_sample 에 있던 기능
  input_tokens = new_input_ids[:, :-1]
  output_tokens = new_input_ids[:, 1:]
  new_attention_mask = new_attention_mask[:, 1:] 
  
  return input_tokens, output_tokens, new_attention_mask

## eos_after_gen
# if False
#  [pad, pad, pad , eos, ~~~~~, eos]
#  [pad, pad, eos ,~~~~~  ~~~~~, eos]
# if True
#  [pad ,eos,  ~~~~~, eos, eos, eos]
#  [eos ,~~~~  ~~~~~, eos, eos, eos]   # and attention mask over latter eos

def ar_sft_loss(self, x0, attention_mask):
  input_tokens, output_tokens, attention_mask = ar_preprocess(self, x0, attention_mask)
  if self.config.backbone == 'llama':
    logprobs = self.backbone(input_tokens).logits.log_softmax(dim=-1)
  else:
    logprobs = self.backbone(input_tokens, None)
  loss = - logprobs.gather(
    -1, output_tokens[:, :, None])[:, :, 0]
  
  nlls = loss * attention_mask
  count = attention_mask.sum()
  batch_nll = nlls.sum()
  token_nll = batch_nll / count
  
  return Loss(loss=token_nll,
            nlls=nlls,
            token_mask=attention_mask)  

## sampling
def topk_decoding(self, next_logits):
  
    topk_val, topk_ind = next_logits.topk(k=self.config.ar.topk_k, dim=-1)
    sampled_pseudo_ind = torch.multinomial(topk_val.exp(), num_samples=1)
    sampled_ind = topk_ind.gather(dim=-1, index=sampled_pseudo_ind)
    return sampled_ind


def nucleus_decoding(self, next_logits):
    prob= next_logits.exp()
    sorted_prob, sorted_ind = torch.sort(prob, descending=True)
    cumulative_prob = torch.cumsum(sorted_prob, dim=-1)
    V = (cumulative_prob > self.config.ar.nucleus_p).int().argmax(dim=-1) + 1 
    column_indices = torch.arange(sorted_prob.size(1)).repeat(sorted_prob.size(0), 1).to(self.device)
    mask = column_indices < V.unsqueeze(-1)
    nucleus_numer = (sorted_prob * mask)
    nucleus_prob = nucleus_numer / nucleus_numer.sum(dim=-1).unsqueeze(-1)
    sampled_pseudo_ind = torch.multinomial(nucleus_prob, num_samples=1)
    sampled_ind = sorted_ind.gather(dim=-1, index=sampled_pseudo_ind)
    return sampled_ind
  
@torch.no_grad()
def _ar_sampler_conditional(self, batch): 
    x = batch['xT'].to(self.device)
    content_len = (x != self.tokenizer.pad_token_id).sum(dim=-1)
    num_cols = x.shape[1]
    batch_indices = torch.arange(num_cols).repeat(x.shape[0], 1).to(self.device)  # [[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]]
    shifted_indices = (batch_indices - self.tokenizer.model_max_length + content_len.view(-1, 1) +0)  % num_cols   # 끝에 패딩 없어야돼서 0
    new_input_ids = torch.gather(x, 1, shifted_indices).to(x.device)
    pad_cut =(new_input_ids==self.tokenizer.pad_token_id).sum(dim=-1).min()
    x = new_input_ids[:, pad_cut:].to(self.device)
    
    num_pred_tokens = self.config.model.length - 1
    if self.config.ar.decoding == 'categorical':
      noise = (torch.distributions.Gumbel(0, 1)
              .sample((x.size(0), num_pred_tokens, self.vocab_size))
              .to(self.device))
    
    for i in range(num_pred_tokens):
      if self.backbone=='llama':
        next_logits = self.forward(x)[:, -1].to(torch.bfloat16)
      else:
        next_logits = self.forward(x, None)[:, -1]  # softmax 돼서 나오는듯
      
      # nucleus decoding
      if self.config.ar.decoding == 'nucleus':
        sampled_ind = nucleus_decoding(self, next_logits)
      elif self.config.ar.decoding == 'topk':
        sampled_ind = topk_decoding(self, next_logits)
      elif self.config.ar.decoding == 'categorical':
        sampled_ind = (next_logits + noise[:, i]).argmax(-1).unsqueeze(-1)
      
      x= torch.concat((x, sampled_ind), dim=-1)
      
    
    finetune_mask = (x != self.mask_index).int()  # query + answer 전부 포함  (diffusion이랑 다름)
      
    return x, finetune_mask


## rdpo


  
def ar_rdpo_preprocess(self, x, attention_mask):
    x = x.to(self.device)
    attention_mask = attention_mask.to(self.device)
    content_len = (x != self.tokenizer.pad_token_id).sum(dim=-1).to(self.device)
    num_cols = x.shape[1]
    batch_indices = torch.arange(num_cols).repeat(x.shape[0], 1).to(self.device)  # [[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]]
    shifted_indices = (batch_indices - self.tokenizer.model_max_length + content_len.view(-1, 1) +0)  % num_cols   # 끝에 패딩 없어야돼서 0
    new_x = torch.gather(x, 1, shifted_indices).to(x.device)
    new_attention_mask = torch.gather(attention_mask, 1, shifted_indices).to(x.device)

    input_new_x = new_x[:, :-1]
    output_new_x = new_x[:, 1:]
    new_attention_mask = new_attention_mask[:, 1:] 
    return input_new_x , output_new_x, new_attention_mask
  
def _forward_pass_AR_finetune_rDPO(self, x0, xT, attention_mask):
  xb, xt, repeat_mask = q_xt_finetune_repeat(self, x0, xT, None) # repeat_mask : repeat_q_xt 에서의 mask 역할
    # xb : rejected label   with repeated patterns
    # ar에서는 xt 가 필요없음.
    
  ## 좌측정렬을 우측정렬로 바꿔주기 
  input_xb, output_xb, repeat_mask = ar_rdpo_preprocess(self, xb, repeat_mask)
  input_x0, output_x0, attention_mask = ar_preprocess(self, x0, attention_mask)  
  
   
  x0_logprobs = self.backbone(input_x0, None)
  xb_logprobs = self.backbone(input_xb, None)
  log_p_w = - x0_logprobs.gather(-1, output_x0[:, :, None])[:, :, 0].squeeze(-1) * attention_mask
  log_p_l = - xb_logprobs.gather(-1, output_xb[:, :, None])[:, :, 0].squeeze(-1) * repeat_mask
  denom_w = attention_mask.sum(dim=-1)
  denom_l = repeat_mask.sum(dim=-1)
  loss = - torch.log(torch.sigmoid( ((self.config.repeat_dpo.beta_w * log_p_w.sum(dim=-1) /denom_w) - (self.config.repeat_dpo.beta_l * log_p_l.sum(dim=-1) /denom_l))- self.config.repeat_dpo.gamma)) - ((self.config.repeat_dpo.beta_a * log_p_w.sum(dim=-1))/denom_w)
  
  loss = loss.sum()
  
  with torch.no_grad():
      history_log_p_w = -torch.log(torch.sigmoid((log_p_w.sum(dim=-1)/denom_w).sum()))
      history_log_p_l = -torch.log(torch.sigmoid((log_p_l.sum(dim=-1)/denom_l).sum()))
  
  return loss, history_log_p_w, history_log_p_l



########### semi-AR  sampling ##########
@torch.no_grad()
def _sample_semi_ar(self, batch, num_steps=None, eps=1e-5, tqdm_disable=True):
  ##### input으로 주어지는 것 #####
  # S = num_steps
  # L = total token length
  # _L = block stride_length

  if num_steps is None:
    S = self.config.sampling.steps
  else:
    S = num_steps
  L = self.tokenizer.model_max_length
  _L = self.config.sampling.semi_ar_stride_length
  
  ######################
  B = int(L / _L)  # block number   i.e.,  num strides
  _S = int(S / B)  # num_steps_per_one_stride
  
  assert   L % _L  ==0
  assert   S % B == 0
  assert  _S >=1

  # dt = (1 - eps) / _S  # _sample_conditional  안에서 구해짐.


  ### preparing  
  n_samples = batch['xT'].size(0)
  ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)
  finetune_mask = (batch['xT'] == self.mask_index).int()  # 직접 생성한거   를 나타내게 될 것
  prompted = (batch['xT'] != self.mask_index).to(self.device) # in case of conditioned generation
  
  for stride in tqdm(range(B + 1), disable=tqdm_disable):
    x , _ = _sample_conditional(self, batch, _S)    

    x = self.forward(x, 0 * ones).argmax(dim=-1) # 0*ones 는 의미 없음
        # 1024 전체에 mask가 많이 남아있음. 그걸 기냥 한방에 다 채우는거임
    
    unmasked = torch.ones_like(x).bool()
    unmasked[:,    _L * (stride +1 ) :] =False
    unmasked = (unmasked | prompted)  # conditioned generation의 경우 prompt 가 _L보다 길 경우 첫번째 stride에서 잘리는 경우가 있음. 이를 막기 위함.
    
    x[~ unmasked] = self.mask_index
    
    if self.config.sampling.eos_fill:  
      eos_found = ((x == self.tokenizer.eos_token_id).cumsum(dim=-1) >=2)
      x[eos_found] = self.tokenizer.eos_token_id     
      
    
    batch['xT']= x  # 여기에 태워서 다시 inference 시킴.
    
  
  return x, finetune_mask