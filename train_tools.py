from finetune_tools import _loss_finetune, _sample_conditional, prior_std_mean_calcul_conditional, _loss_finetune_dpo, _ar_sampler_conditional
from repeat_dpo_tools import _loss_dpo,  rep_4gram_calcul
from collections import OrderedDict
from tqdm import tqdm

import torch


def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
      
    
    # assert (self.config.repeat_dpo.bool==self.config.finetune.bool ==True) ==False    # raise exception if repeat_dpo, finetune is True at the same time
    
    if self.config.repeat_dpo.bool and self.config.finetune.bool:
      losses = _loss_finetune_dpo(self, batch['x0'], batch['xT'], attention_mask, prefix)
    
    elif self.config.repeat_dpo.bool:
      losses = _loss_dpo(self, batch['input_ids'], attention_mask, prefix)
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

    loss = losses.loss
    
    return loss, metrics


def validation_step(self, batch, batch_idx, global_step, rank):
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
        # print(f"prior_std_MSE  {prior_std_MSE},   prior_mean_MSE  {prior_mean_MSE}")
        self.prior_mean_metric.update(prior_mean, torch.ones_like(prior_mean))
        self.prior_mean_MSE_metric.update(prior_mean_MSE, torch.ones_like(prior_mean_MSE))
        self.prior_std_metric.update(prior_std, torch.ones_like(prior_std))
        self.prior_std_MSE_metric.update(prior_std_MSE, torch.ones_like(prior_std_MSE))

      if batch_idx==0:
        if rank == 0 and hasattr(
          self.trainer.logger, 'log_table'):
          # Log the last generated samples
          text_samples = text_samples[: self.config.sampling.num_sample_log]
          # print(self.global_step)
          self.trainer.logger.log_table(
            key=f'samples@global_step{global_step}_{rank}',
            columns=['Generated Samples'],
            data=[[s] for s in text_samples])
    return _compute_loss(self, batch, prefix='val')
  
  
def on_validation_epoch_end(self, global_step, epoch):
  
  # if self.config.eval.compute_prior_std_mean:
  gen_ppl = self.gen_ppl_metric.compute()
  prior_mean = self.prior_mean_metric.compute()
  prior_mean_MSE = self.prior_mean_MSE_metric.compute()
  prior_std = self.prior_std_metric.compute()
  prior_std_MSE = self.prior_std_MSE_metric.compute()
  logged_metrics = {'prior_mean':prior_mean,
                    'prior_mean_MSE':prior_mean_MSE,
                    'prior_std':prior_std,
                    'prior_std_MSE':prior_std_MSE,
                    'gen_ppl':gen_ppl,
                    'global_step':global_step}
  # print(logged_metrics)
  self.trainer.logger.log_metrics(logged_metrics)
  
  self.prior_mean_metric.reset()
  self.prior_mean_MSE_metric.reset()
  self.prior_std_metric.reset()
  self.prior_std_MSE_metric.reset()
    
  self.gen_ppl_metric.reset()
    
    
  if self.ema:
    self.ema.restore(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()))




def eval_step(config, ddp_model,rank, world_size, dataloader_valid, global_step, epoch, ):
        ddp_model.module.eval()
        # torch.cuda.empty_cache()
        if rank==0:
            print("eval 시작")
        with torch.no_grad():
            ddp_model.module.backbone.eval()
            ddp_model.module.noise.eval()
            ddp_model.module.valid_metrics.nll.mean_value == 0
            ddp_model.module.valid_metrics.nll.weight == 0
            
            ddp_model.module.on_validation_epoch_start()
            total_steps = len(dataloader_valid)
            loss_list = []
            tqdm_disable = False if rank==0 else True
            for i, batch_data in tqdm(enumerate(dataloader_valid), disable=tqdm_disable, total=total_steps):
              batch_data = {k: v.to(rank, non_blocking=True) for k, v in batch_data.items()}
              loss, metrics = validation_step(ddp_model.module, batch_data, i, global_step, rank)
              # ddp_model.module.validation_step( batch_data, i)
            loss_list.append(loss)
            
            metric_values = metrics.compute()
            logged_metrics = {f"{key}": value.item() for key, value in metric_values.items()}
            logged_metrics['global_step'] = global_step
            logged_metrics['epoch'] = epoch
            with torch.no_grad():
              loss_mean = torch.tensor(loss_list).mean().item()
            logged_metrics['val/loss'] = loss_mean 
            ddp_model.module.trainer.logger.log_metrics(logged_metrics)
            on_validation_epoch_end(ddp_model.module, global_step, epoch)
            ddp_model.module.valid_metrics.reset()
            
            
            if rank==0:
                # self.model
                state_dict = OrderedDict()
                for name, param in ddp_model.module.named_parameters():
                    if param.requires_grad:
                        state_dict[name] = param.clone().detach().cpu()

                ## save
                # if args.save_on_eval:
                result_dic = {'config': {** config},
                }
                result_dic['config']['world_size'] = world_size
                result_dic['state_dict']=state_dict

                torch.save(result_dic, f'/home/ybseo/data/mdlm/outputs_custom/{config.wandb.name}/cp_ep{epoch}_gbstep{global_step}.ckpt')

                