
import torch
import utils
import random
from dataclasses import dataclass
from nltk import ngrams
from collections import Counter
import torch.nn.functional as F

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor
  loss_w: torch.FloatTensor
  loss_l: torch.FloatTensor


def in_mask_repeat(model, i, x_t0, rand_val, min_phase_len=15):
    left, right = rand_val[i]
    mask_len = right - left
    side = random.choice(['left','right'])
    if side=='left':
        context = x_t0[i][1:left]
        phase_len = torch.randint(1, min(mask_len +1 ,len(context) +1, min_phase_len), (1,))
        repeat = context[-phase_len:]
    else:
        context = x_t0[i][right:-1]
        phase_len = torch.randint(1, min(mask_len +1 ,len(context) +1, min_phase_len), (1,))
        repeat = context[:phase_len]
            
    for mask_i in range(left,right):
        x_t0[i][mask_i] = repeat[mask_i % len(repeat) ]
        
def out_mask_repeat(model, i, x_t0, rand_val, min_phase_len=15):
    left, right = rand_val[i]
    context = x_t0[i][left : right]
    context_len = right - left
    
    # left side
    phase_len = torch.randint(1, min(context_len +1 , left +1, min_phase_len), (1,)) #  < min  이기때문에, context_len +1 일 경우 phase_len 의 최대값은 context_len
    repeat = context[:phase_len]
    rand_permute = torch.randint(0, len(repeat),(1,))
    for mask_i in range(1,left): # 왼쪽 1칸 남겨놓기 위해 1부터 함. 
    # for mask_i in range(0,left): # eos 토큰까지 repeat
        x_t0[i][mask_i ] = repeat[(mask_i + rand_permute) % len(repeat) ]
    
    #right side
    phase_len = torch.randint(1, min(context_len +1, right +1, min_phase_len), (1,))
    repeat = context[-phase_len:]
    rand_permute = torch.randint(0, len(repeat),(1,))

    for mask_i in range(right, x_t0.size(-1)-1):
    # for mask_i in range(right, x_t0.size(-1)): # eos 토큰까지 repeat
        x_t0[i][mask_i] = repeat[(mask_i + rand_permute) % len(repeat) ]


def repeat_q_xt(model, x0):
    
    in_or_out = (torch.randint(low=0, high=10, size=(1,x0.size(0))) >=5 )[0] # shape: [batch_size] 
    # e.g, [True , False, False]
    # if True (in) :   [word, word, mask, mask, word, word]
    # if False (out) :   [mask, mask, word, word, mask, mask]
    
    
    
    rand_val = []
    for ioo in in_or_out:
        if ioo:
            low, high = 5, x0.size(-1) -5  # 좌우 end_of_text 에 토큰 1씩 더 남기고
            a = torch.randperm(high - low)[:2] + low
            a= a[a.argsort()]
            rand_val.append(a)
        else:
            low, high = 5, x0.size(-1) -5  # 좌우 end_of_text 에 토큰 1씩 더 남기고
            a = torch.randperm(high - low)[:2] + low
            a= a[a.argsort()]
            rand_val.append(a)
    rand_val = torch.stack(rand_val)  # shape: [batch_size x 2]
    # e.g, [[5, 11], [3,7]]   
    # 1st and 2nd elements indicates the boundary index between mask and words.
    
    # mask tensor, masked xt
    pallet = torch.arange(0,x0.size(-1)).repeat(x0.size(0),1)
    left = pallet < rand_val[:,0].unsqueeze(-1)  
    right = pallet >= rand_val[:,1].unsqueeze(-1) 
    mask = ((left | right) ^ in_or_out.unsqueeze(-1)).to(x0.device)
    xt = x0.clone()
    xt[mask] = model.mask_index  
    
    xb = xt.clone()
    for i in range(x0.size(0)):
        if in_or_out[i]:
            in_mask_repeat(model, i, xb, rand_val, model.config.repeat_dpo.min_phase_len)
        else:
            out_mask_repeat(model, i, xb, rand_val, model.config.repeat_dpo.min_phase_len)
        
    return xt, xb, mask


def _forward_pass_diffusion_repeat_DPO(self, x0, attention_mask): # diffusion은 여기서 주로 함.
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
    ##### move_chance, t   사실상 쓰이지 않음 #####



    xt, xb, mask = repeat_q_xt(self, x0) #  transition matrix ********. 
    # xt : masked
    # xb : rejected label   with repeated patterns

      
    # loss function

    model_output = self.forward(xt, unet_conditioning) # logprobability (batch X sentence length X vocab size)  # unet_conditioning is not used actually

    utils.print_nans(model_output, 'model_output')

    if self.parameterization in ['d3pm', 'subs']:
        
        # yw > yl
        log_p_w = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)  * mask
        log_p_l = torch.gather(
            input=model_output,
            dim=-1,
            index=xb[:, :, None]).squeeze(-1) * mask
        
        if self.config.repeat_dpo.denom=='mask_len':
            denom = mask.sum(dim=-1)
        elif self.config.repeat_dpo.denom=='seq_len':
            denom = mask.size(dim=-1)
        else:
            raise Exception("!!! choose repeat_dpo denom type")
        
        log_p_l[:,0] = 0  #
        log_p_l[:,-1] = 0
        
        loss = - torch.log(torch.sigmoid( (((self.config.repeat_dpo.beta_w * log_p_w.sum(dim=-1)) - (self.config.repeat_dpo.beta_l * log_p_l.sum(dim=-1)))  / denom)- self.config.repeat_dpo.gamma)) - ((self.config.repeat_dpo.beta_a * log_p_w.sum(dim=-1))/denom)
        
        loss = loss.sum()
        
        with torch.no_grad():
            history_log_p_w = -torch.log(torch.sigmoid((log_p_w.sum(dim=-1)/denom).sum()))
            history_log_p_l = -torch.log(torch.sigmoid((log_p_l.sum(dim=-1)/denom).sum()))
        
        return loss, history_log_p_w, history_log_p_l
    else:
        raise Exception("!!!!! parameterization not yet prepared !!!!!!!")


def _loss_dpo(self, x0, attention_mask, prefix=None):
    (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(x0, attention_mask)
    
    loss, history_log_p_w, history_log_p_l  = _forward_pass_diffusion_repeat_DPO(self, input_tokens, attention_mask)
    
        
    
    return Loss(loss=loss, token_mask=attention_mask, nlls=torch.ones_like(attention_mask, device=attention_mask.device)* loss.detach(), loss_w=history_log_p_w, loss_l=history_log_p_l )


### sampling

def _sample_categorical_topk_gumbel(categorical_probs , k, model=None, temperature=1):
    topk_val, topk_ind =categorical_probs.topk(k, dim=-1)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(topk_val) + 1e-10).log()) 
    gumbel_norm = gumbel_norm ** temperature
    argmax_in_topk = (topk_val / gumbel_norm).argmax(dim=-1)
   

    result = topk_ind.gather(dim=-1 , index=argmax_in_topk.unsqueeze(-1)).squeeze(-1)
    return result

def _sample_categorical(categorical_probs , model=None, temperature=1):  
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    gumbel_norm = gumbel_norm ** temperature
    return (categorical_probs / gumbel_norm).argmax(dim=-1)

def _ddpm_topk_update(self, x, t, dt):
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
    p_x0 = log_p_x0.exp()
    


    if type(log_p_x0)==tuple:
        log_p_x0= log_p_x0[0]

    assert move_chance_t.ndim == log_p_x0.ndim
    
    #####global norm 
    masked = x == self.mask_index
    masked_p_x0 = p_x0[masked]
    
    
    global_norm = masked_p_x0.sum() / masked_p_x0.topk(k=self.config.sampling.topk_k,dim=-1).values.sum()

    q_xs = p_x0 * (move_chance_t - move_chance_s)  * global_norm

    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical_topk_gumbel(q_xs, self.config.sampling.topk_k, self , self.config.sampling.temperature)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x 


def _ddpm_topk_caching_update_old(self, x, t, dt, p_x0=None):
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
    
    masked = x == self.mask_index
    masked_p_x0 = p_x0[masked]
    global_norm = masked_p_x0.sum() / masked_p_x0.topk(k=self.config.sampling.topk_k,dim=-1).values.sum()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)  * global_norm
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs) 
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

def _ddpm_topk_caching_update(self, x, t, dt, p_x0=None):
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
    
    if p_x0 ==None:
      log_p_x0 = self.forward(x, unet_conditioning)
      p_x0 = log_p_x0.exp()

      assert move_chance_t.ndim == log_p_x0.ndim
      
      #####global norm   
      masked = x == self.mask_index
      masked_p_x0 = p_x0[masked]
      
      
      global_norm = masked_p_x0.sum() / masked_p_x0.topk(k=self.config.sampling.topk_k,dim=-1).values.sum()

      p_x0 = p_x0 * global_norm

    q_xs = p_x0 * (move_chance_t - move_chance_s) 

    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical_topk_gumbel(q_xs, self.config.sampling.topk_k, self, self.config.sampling.temperature)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    x_next = copy_flag * x + (1 - copy_flag) * _x
    return  p_x0 , x_next

def _ddpm_convolution_update(self, x, t, dt):
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
    p_x0 = log_p_x0.exp()

    assert move_chance_t.ndim == log_p_x0.ndim
    if self.config.sampling.topk_k >0: # topk
        k= self.config.sampling.topk_k  + 1   # index for mask
    else:  # categorical
        k= self.tokenizer.vocab_size   # 50257
    topk_p_x0, topk_p_x0_ind = p_x0.topk(k= k, dim=-1)
    masked = x == self.mask_index
    masked_topk_p_x0 = topk_p_x0[masked]
    masked_p_x0 = p_x0[masked]    #  sum^N sum^K  p(x_ij)
    
    ## convolutional decoding
    # s_i
    unmasked = ~masked   # [batch, len]
    input_seq = unmasked.float().view(masked.size(0), 1, -1).to('cuda')  # [batch, 1, len]
    conv0 = F.conv1d(input_seq, self.backbone.kernel, stride=1, padding=self.backbone.padding_size.item()).squeeze(1)  # [batch, len]
    # max: window size. If there's a condition on the left, the end of the condition is the center of the window (e.g., center is 6 for window size 11)
    normed_conv0 = conv0  # Not normalizing here; scaling is handled via conv_mult
    conved_s = F.tanh(normed_conv0 * self.config.sampling.conv_mult)  # Normalization term s  [1, len]
    s_norm_constant = (masked_p_x0.sum() / (masked_topk_p_x0 * conved_s[masked].unsqueeze(-1)).sum())  # Scalar (step-wise variable)
    conved_s_normed = conved_s * s_norm_constant  # [1, len]
    p_x0 = topk_p_x0 * conved_s_normed.unsqueeze(-1)  # [1, len, k] 
    # The normalizing term is j (candidate rank) agnostic, so unsqueeze is applied.
    # Although normalization is computed over masked tokens, it is broadcast over all p_x0 -> this does not affect total probability mass because it will be overwritten by copy_flag later

    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, -1] = move_chance_s[:, :, 0]  # Since top-k is used, shape is k+1; the last column corresponds to [MASK]
    topk_p_x0_ind[:, :, -1] = self.mask_index  # Also update the index accordingly
    sampled_ind = _sample_categorical(q_xs, None, self.config.sampling.temperature)
    _x = torch.gather(topk_p_x0_ind, -1, sampled_ind.unsqueeze(-1)).squeeze(-1)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x


def _ddpm_convolution_caching_update(self, x, t, dt, p_x0=None):
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
    
    if p_x0 == None:
        log_p_x0 = self.forward(x, unet_conditioning)
        p_x0 = log_p_x0.exp()

        assert move_chance_t.ndim == log_p_x0.ndim
        if self.config.sampling.topk_k >0: # topk
            k= self.config.sampling.topk_k  + 1   # index for mask
        else:  # categorical
            k= self.tokenizer.vocab_size   # 50257
        topk_p_x0, topk_p_x0_ind = p_x0.topk(k= k, dim=-1)
        masked = x == self.mask_index
        masked_topk_p_x0 = topk_p_x0[masked]
        masked_p_x0 = p_x0[masked]    #  sum^N sum^K  p(x_ij)
        
        ## convolutional decoding
        # s_i
        unmasked = ~masked   # [batch, len]
        input_seq = unmasked.float().view(masked.size(0), 1, -1).to('cuda')  # [batch, 1, len]
        conv0 = F.conv1d(input_seq, self.backbone.kernel, stride=1, padding=self.backbone.padding_size.item()).squeeze(1)  # [batch, len]
        # max: window size. If there is a conditioning segment on the left, the end of that segment aligns with the center of the window (e.g., position 6 for window size 11)

        # normed_conv0 = conv0 / conv0.max()  # Normalize so that max becomes 1 (was fixed to 1 previously)
        normed_conv0 = conv0  # We skip normalization; scaling will be handled entirely via conv_mult
        conved_s = F.tanh(normed_conv0 * self.config.sampling.conv_mult)  # Normalization term s [1, len]
        # conved_s = F.sigmoid(normed_conv0 * self.config.sampling.conv_mult)  # (Alternative) normalization term s [1, len]

        # The numerator is sum^V, so we use masked_p_x0 (not masked_topk_p_x0) for computing the normalizing constant
        s_norm_constant = (
            masked_p_x0.sum() / (masked_topk_p_x0 * conved_s[masked].unsqueeze(-1)).sum()
        )  # Scalar (step-dependent variable)

        conved_s_normed = conved_s * s_norm_constant  # [1, len]
        p_x0 = topk_p_x0 * conved_s_normed.unsqueeze(-1)  # [1, len, k]
        # The normalization term is agnostic to candidate rank j, so we apply unsqueeze
        # Although the normalizing constant is computed on masked positions, it's broadcast to the entire p_x0
        # -> This has no effect on the final probability mass because it will be overridden by copy_flag later


    q_xs = p_x0 * (move_chance_t - move_chance_s)
    # q_xs.shape
    # q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    q_xs[:, :, -1] = move_chance_s[:, :, 0]  # Using top-k candidates, shape becomes k+1; the last column corresponds to [MASK]
    topk_p_x0_ind[:, :, -1] = self.mask_index  # Update the index for [MASK] accordingly
    sampled_ind = _sample_categorical(q_xs, None)
    _x = torch.gather(topk_p_x0_ind, -1, sampled_ind.unsqueeze(-1)).squeeze(-1)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x



###### eval metrics






def prior_std_mean_calcul(model, samples):
    prior_mean = model.backbone.token_prior_for_eval[samples].log().mean(dim=-1)   # log prior 의 mean .  [batchsize]
    prior_std = (model.backbone.token_prior_for_eval[samples]*1000).std(dim=-1)   # log prior 의 std .  [batchsize]
    prior_mean_MSE = (prior_mean - model.backbone.token_prior_logs)**2
    prior_std_MSE = (prior_std - model.backbone.token_prior_stds)**2
    return prior_mean, prior_mean_MSE , prior_std, prior_std_MSE

