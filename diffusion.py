import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor
import torch.nn as nn 

import dataloader
import models
import noise_schedule
import utils
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from timeit import default_timer
from finetune_tools import _loss_finetune, _sample_conditional, prior_std_mean_calcul_conditional, _loss_finetune_r2ft, _ar_sampler_conditional
from r2ft_tools import _loss_r2ft, _ddpm_topk_update, rep_4gram_calcul, prior_std_mean_calcul, _ddpm_convolution_update, _ddpm_topk_caching_update


LOG2 = math.log(2)


def _sample_categorical(categorical_probs , model=None):  # gumbel softmax 로 categorical sampling 을 대신한것*****
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
    
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)

class Norm2(NLL):
  def compute(self) -> Tensor:
    """Computes the Norm2.
    mean_value :  tensor list of (a-b)**2
    Returns:
     2Norm
    """
    return self.mean_value.sum().sqrt()

class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.history_buffer = defaultdict(list)  # 관찰하기 위함

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
      
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dit_positional': #
      self.backbone = models.dit.DIT_positional(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':

        self.backbone = models.autoregressive.AR(
          self.config,
          vocab_size=self.vocab_size,
          mask_index=self.mask_index)
    elif self.config.backbone == 'llama':
        self.backbone = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16)
        self.tokenizer.model_max_length= config.model_max_length
        self.tokenizer.pad_token_id= 128009  # <|eot_id|>
        self.mask_index = self.tokenizer.pad_token_id
        
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    elif self.config.backbone == 'llada':
      self.backbone = transformers.AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
      self.mask_index = tokenizer.encode("<|mdm_mask|>")[0] 
      self.tokenizer.pad_token_id =  tokenizer.encode("<|mdm_mask|>")[0]
      self.tokenizer.model_max_length= config.model_max_length
    elif self.config.backbone == 'llada_inst':
      self.backbone = transformers.AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16)
      self.mask_index = tokenizer.encode("<|mdm_mask|>")[0] 
      self.tokenizer.pad_token_id =  tokenizer.encode("<|mdm_mask|>")[0]
      self.tokenizer.model_max_length= config.model_max_length
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')
      
    if self.config.lora.bool :
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
            ],
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
    self.T = self.config.T
    self.subs_masking = self.config.subs_masking
    
    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics_simple = torchmetrics.MetricCollection({
      'nll': NLL()})
    metrics.set_dtype(torch.float64)
    metrics_simple.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.valid_dpo_w_metrics = metrics_simple.clone(prefix='val/_dpo_w_')
    self.valid_dpo_l_metrics = metrics_simple.clone(prefix='val/_dpo_l_')
    self.test_metrics = metrics.clone(prefix='test/')
    
    # metric : repeat, prior var, prior mean
    self.rep_4gram_metric = NLL()  # repetition metric
    self.prior_std_metric = NLL()
    self.prior_mean_metric = NLL() 
    self.prior_std_MSE_metric = NLL()
    self.prior_mean_MSE_metric = NLL() 
    
    if config.eval.compute_prior_std_mean:
      token_prior = torch.load(f'{config.util_data_dir}/all_prior_{self.config.prior_type}.pt') # default tfdf
      logs = torch.load(f'{config.util_data_dir}/{config.data.train}/all_data_logs.pt').median() /512  #
      stds = torch.load(f'{config.util_data_dir}/{config.data.train}/all_data_stds.pt').median() #
      self.backbone.register_buffer('token_prior_for_eval', token_prior , persistent=False)
      self.backbone.register_buffer('token_prior_logs', logs , persistent=False)
      self.backbone.register_buffer('token_prior_stds', stds , persistent=False)

    if 'convolution' in self.sampler:
      kernel_size = self.config.sampling.convolution_kernel_size   # window_size
      if kernel_size % 2 !=1:  # must be odd number
        kernel_size +=1
      kernel = torch.ones(kernel_size)  
      kernel = kernel.view(1, 1, -1)   # [1,1, window_size]
      padding_size = torch.tensor((kernel_size -1 ) // 2 )
      self.backbone.register_buffer('kernel' , kernel, persistent=False)
      self.backbone.register_buffer('padding_size' , padding_size, persistent=False)
    
    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()
    
      

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    # if self.parameterization == 'd3pm':  #  continuous d3pm 
      # assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.config.save_weight_only: #cppm
      return 0
    
    # if self.backbone== 'llada': # 어차피 이리로 안옴
    #   del checkpoint['state_dict']
    #   dic = OrderedDict()
    #   for name, params in self.backbone.named_parameters():
    #       if params.requires_grad==True:
    #           dic[name] =params.detach().cpu()
    #   checkpoint['state_dict'] = dic
      
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  
    
  
  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity   # -> probability = 0 으로 만듦
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # log softmax 와 같음. 이제 logits은 log probability임.
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)  # mask token이 아닌 position

    logits[unmasked_indices] = self.neg_infinity   # mask token 이 아닌것의 log probability는 모두 -inf , probability = 0
    # logits[unmasked_indices] = 0
    
    logits[unmasked_indices, xt[unmasked_indices]] = 0  # unmasked 에서 그 토큰 index는  log prob=0 ,   probability = 1   
    
    
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking: # default : False
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,   # just log softmax
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:  # default
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, attention_mask=None):
    """Returns log score."""
    sigma = self._process_sigma(sigma)  # time_conditioning=False 가 디폴트. 그러면 sigma=0 이 됨
    # with torch.cuda.amp.autocast(dtype=torch.float32):# 이게 시간이 오지게 걸림** 아래서 각각 float32로 지정하기로
    if 'llada' in self.config.backbone:
      logits = self.backbone(x).logits.to(torch.float32)  # to 안하면 bfloat인데, 속도가 더 줄진 않음.
    elif self.config.backbone=='llama':
      logits = self.backbone(x).logits.to(torch.float32)  # to 안하면 bfloat인데, 속도가 더 줄진 않음.
    else:
      logits = self.backbone(x, sigma).to(torch.float32) # 여기서부터 not deterministic
        
      
    if self.config.backbone=='dit_positional': #cppm.   backbone 에서  (x_cl, x_p) 가 나온다는 뜻
      logits, logits_p = logits
    
    # return logits
    # ## repetition penalty
    # if self.config.sampling.repetition_penalty > 1.0 : # 1이면 없는것임임
    #   penalty = float(self.config.sampling.repetition_penalty)
    #   x_expanded = x.unsqueeze(1).expand(-1, x.size(-1), -1)
    #   score = torch.gather(logits, dim=2, index=x_expanded.to(torch.int64))
    #   # score = torch.where(score < 0, score * penalty ,  score / penalty)
    #   score /= penalty
    #   logits = logits.scatter(-1, x_expanded.to(torch.int64), score)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      # sampling할떄 이리로 들어옴
      return self._d3pm_parameterization(logits=logits)
    
    
    else:
      return logits # 여기로 바로나가는 경우는 거의 없을듯. 

  def _d3pm_loss(self, model_output, xt, x0, t):  # model_output : log probability
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
      
    
    
    # assert (self.config.r2ft.bool==self.config.finetune.bool ==True) ==False    # raise exception if r2ft, finetune is True at the same time
    
    if self.config.r2ft.bool and self.config.finetune.bool:
      losses = _loss_finetune_r2ft(self, batch['x0'], batch['xT'], attention_mask, prefix)
    
    elif self.config.r2ft.bool:
      losses = _loss_r2ft(self, batch['input_ids'], attention_mask, prefix)
    elif self.config.finetune.bool:
      losses = _loss_finetune(self, batch['x0'], batch['xT'], attention_mask, prefix)
    else:
      losses = self._loss(batch['input_ids'], attention_mask, prefix) # prefix 추가. log 위함
      
    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')


    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                    sync_dist=True)
    loss = losses.loss
    
    return loss

  def on_train_epoch_start(self):  # lightning innate func.
    self.backbone.train() 
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    ############ finetune 일 경우 여기서 다함 ######## (batch가 필요해서)
    # batch_idx =8 이후부터는 compute_loss만 함.
    if self.config.finetune.bool and batch_idx <8: # on val epoch end 에 할 수 없어서 여기로 빼온것. # 
      # print(batch_idx)
      if self.parameterization=='ar':
        samples, finetune_mask = _ar_sampler_conditional(self, batch)
      else:
        samples, finetune_mask = _sample_conditional(self,batch)
      #finetune_mask : 직접 작성한 부분 = 1
      # val_eos_off=True 이면:  직접 작성한 부분 & eos 가 아닌 부분 = 1
      text_samples = self.tokenizer.batch_decode(samples)
      if self.config.eval.compute_generative_perplexity:
        # self.compute_generative_perplexity(text_samples, finetune_mask = finetune_mask)
        self.compute_generative_perplexity(text_samples, finetune_mask = finetune_mask, retokenize=self.config.eval.retokenize) # False 
        # retokenize : default is True. But should be False when eval on finetuning. finetune_mask 를 적용해야 하기 때문 ********

      if self.config.eval.compute_rep_4gram:
        rep_4gram = rep_4gram_calcul(samples) # input: torch.tensor, output: list
        self.rep_4gram_metric.update(rep_4gram, torch.ones_like(rep_4gram))
      if self.config.eval.compute_prior_std_mean:
        prior_mean, prior_mean_MSE , prior_std, prior_std_MSE= prior_std_mean_calcul_conditional(self, samples, finetune_mask)
        self.prior_mean_metric.update(prior_mean, torch.ones_like(prior_mean))
        self.prior_mean_MSE_metric.update(prior_mean_MSE, torch.ones_like(prior_mean_MSE))
        self.prior_std_metric.update(prior_std, torch.ones_like(prior_std))
        self.prior_std_MSE_metric.update(prior_std_MSE, torch.ones_like(prior_std_MSE))

      if batch_idx==0:
        if self.trainer.global_rank == 0 and hasattr(
          self.trainer.logger, 'log_table'):
          # Log the last generated samples
          text_samples = text_samples[: self.config.sampling.num_sample_log]
          self.trainer.logger.log_table(
            key=f'samples@global_step{self.global_step}',
            columns=['Generated Samples'],
            data=[[s] for s in text_samples])
        

    ############################################
    
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples): # finetune이면 'validation_step()' 에서 다 함.
        #  and not self.parameterization == 'ar'): # 원래 있었는데 ar SFT때문에 지움. 'ar' pretrain할때는 문제가 생길수도 있을듯.
      # TODO(justin): implement sampling and kv cache for AR
      if not self.config.finetune.bool:
        samples, text_samples = None, None
        for _ in range(
          self.config.sampling.num_sample_batches):
          samples = self._sample()
          # Decode the samples to be re-tokenized by eval model
          text_samples = self.tokenizer.batch_decode(samples)
          
          if self.config.eval.compute_generative_perplexity:
            self.compute_generative_perplexity(text_samples)
          
          if self.config.eval.compute_rep_4gram:
            rep_4gram = rep_4gram_calcul(samples) # input: torch.tensor, output: list
            self.rep_4gram_metric.update(rep_4gram, torch.ones_like(rep_4gram))
            
          if self.config.eval.compute_prior_std_mean:
            prior_mean, prior_mean_MSE , prior_std, prior_std_MSE= prior_std_mean_calcul(self, samples)
            self.prior_mean_metric.update(prior_mean, torch.ones_like(prior_mean))
            self.prior_mean_MSE_metric.update(prior_mean_MSE, torch.ones_like(prior_mean_MSE))
            self.prior_std_metric.update(prior_std, torch.ones_like(prior_std))
            self.prior_std_MSE_metric.update(prior_std_MSE, torch.ones_like(prior_std_MSE))
            
            
        if self.trainer.global_rank == 0 and hasattr(
          self.trainer.logger, 'log_table'):
          # Log the last generated samples
          text_samples = text_samples[
            : self.config.sampling.num_sample_log]
          self.trainer.logger.log_table(
            key=f'samples@global_step{self.global_step}',
            columns=['Generated Samples'],
            data=[[s] for s in text_samples])
      else:  # finetune.bool
        pass
      
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
      if self.config.eval.compute_rep_4gram:
        self.log('val/rep_4gram', self.rep_4gram_metric, on_epoch=True, on_step=False, sync_dist=True)
      if self.config.eval.compute_prior_std_mean:
        self.log('val/prior_mean', self.prior_mean_metric, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val/prior_mean_MSE', self.prior_mean_MSE_metric, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val/prior_std', self.prior_std_metric, on_epoch=True, on_step=False, sync_dist=True)
        self.log('val/prior_std_MSE', self.prior_std_MSE_metric, on_epoch=True, on_step=False, sync_dist=True)
        
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None,
    finetune_mask=None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    
    # if self.config.finetune.bool: # finetune_mask : 직접 작성하고  eos 가 아닌 것 =1
      # assert not retokenize  # retokeniz이면 토큰 숫자가 달라져서 attn_mask 를 적용할 수 없음. retokenize 안하면 정확성이 약간 줄어들지만 어쩔수없음.
      # attn_mask = attn_mask * finetune_mask
    
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance): ## transition matrix 인듯****** 
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    
    #### class transition
    if self.config.class_noise_type== 'absorbing':
      move_indices = torch.rand(  # 원래 이쪽.
        * x.shape, device=x.device) < move_chance   # move_chance=t .  t->1 일수록 mc->1
      xt_cl = torch.where(move_indices, self.mask_index, x)  # x에서 move_chance 이하인건 마스크로 바꿔줌 ****
      return xt_cl
    
    

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    
    log_p_x0 = self.forward(x, unet_conditioning)
    if type(log_p_x0)==tuple:
      log_p_x0= log_p_x0[0]
    
    # predicted = log_p_x0.argmax(dim=-1)
    # print(f"{self.tokenizer.convert_ids_to_tokens(predicted[0])}")
    
    # print(f"{log_p_x0.max(dim=-1).values.exp().detach().cpu()}")
    # self.history_buffer['candidate'].append(log_p_x0.exp().detach().cpu().topk(k=20))
    
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)  # 원상복구 뒤 t~s 노이즈를 추가 (class transition 맞춤형) .  t-s = dt , 일정함.
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    
    _x = _sample_categorical(q_xs, self)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x
  


  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(   # [num_steps]   1 ~ eps
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(   # timestep 중 하나를 batchsize만큼 키움
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_topk':
        x = _ddpm_topk_update(self, x, t, dt)
      elif self.sampler == 'ddpm_convolution':
        x = _ddpm_convolution_update(self, x, t, dt)
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
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
     
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length: # 글 길이가 모델 길이보다 길면 -> 모델 길이에 맞게 자르기
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else: # 여기로 옴.(별일 없으면 그냥 지나치는듯)
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0, attention_mask): # diffusion은 여기서 주로 함.
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
      move_chance = 1 - torch.exp(-sigma[:, None]) # 풀어보면 그냥 t 그자체임 ***    # sigma = -log(1-t)  # 근데 shape 이 다른듯 (1, len(t))
      

    if self.config.forward_type == 'cppm':  # cppm
      xt, xt_cl, xt_p = self.q_xt_cppm(x0, move_chance)   
    elif self.config.forward_type == 'ddpm':
      if self.config.class_noise_type == 'uniform':
        xt, uniform_change_mask = self.q_xt(x0, move_chance)
      else:
        xt = self.q_xt(x0, move_chance) # 이부분이 transition matrix ********. 
    else:
      raise Exception("!!!!! Unknown forward type !!!!")

      
    # xt 가 길이가 확장됐다면 그만큼 x0에 패딩을 넣어줌
       
    model_output = self.forward(xt, unet_conditioning, attention_mask) # logprobability (batch X sentence length X vocab size)
    utils.print_nans(model_output, 'model_output')

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


  def _loss(self, x0, attention_mask, prefix=None):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
      
    else:  # diffusion은 여기로
      loss = self._forward_pass_diffusion(input_tokens, attention_mask) 
    
          
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

    
  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad # this is semi-AR code from MDLM work. We implement new one in 'finetune_tools.py'
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths
