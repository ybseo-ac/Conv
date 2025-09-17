import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, prompt_original=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    if prompt_original:
        x = prompt.clone()
    else:
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x



@torch.no_grad()
def generate_semiar(model,tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0., mask_id=126336, k=1, conv_mult=1,eos_fill=False, decode_type='', prompt_original=False):
  ##### input으로 주어지는 것 #####
  # S = num_steps
  # L = total token length
  # _L = block stride_length

  S = steps
  L = gen_length
  _L = block_length

  ######################
  B = int(L / _L)  # block number   i.e.,  num strides
  _S = int(S / B)  # num_steps_per_one_stride

  assert   L % _L  ==0
  assert   S % B == 0
  assert  _S >=1

  # dt = (1 - eps) / _S  # _sample_conditional  안에서 구해짐.

  if prompt_original:
    x = prompt.clone()
  else:
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
  ### preparing  
  n_samples = x.size(0)
  prompted = (x!= mask_id).to(model.device) # in case of conditioned generation  # 이것도 이상해 ;
  prompted_sum = prompted.sum(dim=-1)
  x_history =[]

  for stride in range(B ):
    x  = generate_yb(model,tokenizer, x, steps=_S, gen_length=L, block_length=32, temperature=temperature, mask_id=126336, k=1, conv_mult=1,eos_fill=False, decode_type='topk', semi_ar=True)   # block_length 아무거나 해도 됨

    # x = model(x).logits.argmax(dim=-1) # 0*ones 는 의미 없음
        # 1024 전체에 mask가 많이 남아있음. 그걸 기냥 한방에 다 채우는거임

    unmasked = torch.ones_like(x).bool()
    unmasked[:,  prompted_sum +  _L * (stride +1 ) :] =False
    unmasked = (unmasked | prompted)  # conditioned generation의 경우 prompt 가 _L보다 길 경우 첫번째 stride에서 잘리는 경우가 있음. 이를 막기 위함.

    x[~ unmasked] = mask_id

    # if self.config.sampling.eos_fill:  
    #     eos_found = ((x == self.tokenizer.eos_token_id).cumsum(dim=-1) >=2)
    #     x[eos_found] = self.tokenizer.eos_token_id     
  return x        


@ torch.no_grad()
def generate_llada_convolution(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,             cfg_scale=0., remasking='low_confidence', mask_id=126336, conv_mult=1, prompt_original=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    kernel_size = block_length  ########
    if kernel_size % 2 !=1: 
        kernel_size +=1
    kernel = torch.ones(kernel_size).to(model.device)
    kernel = kernel.view(1, 1, -1)   # [1,1, window_size]
    padding_size = torch.tensor((kernel_size -1 ) // 2 ).to(model.device)
    
    if prompt_original:
        x = prompt.clone()
    else:
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    block_mask_index = (x[:, prompt.shape[1]  : ]  == mask_id)
    num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
    for i in range(steps):
        mask_index = (x == mask_id)
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

        if remasking == 'low_confidence':
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)


        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
        ##### convolution  ------------------------------

        unmasked = ~mask_index
        topk_p_x0 = x0_p.unsqueeze(-1)   # 여기선 항상  k=1인 셈임. 위에서 top 1을 골라서 내려온게 x0이기 때문. 내코드 conv 와 val name 과 dim 맞춰줌.
        topk_p_x0_ind = x0.unsqueeze(-1)
        # masked_p_x0 = x0_p[mask_index]
        # masked_topk_p_x0 = topk_p_x0[masked]
        input_seq = unmasked.float().view(mask_index.size(0),1,-1).to('cuda')  # [batch, 1 , len]
        conv0 = F.conv1d(input_seq, kernel, stride=1, padding=padding_size.item() ).squeeze(1) # [batch,len] # max : window size.   / 좌측 condition이 있을때, condition 끝 값: window size의 중간값. (e.g., 11 중 6)

        normed_conv0 = conv0
        conved_s = F.tanh(normed_conv0 * conv_mult) # normalization term s  [1, len]
        # s_norm_constant = (masked_p_x0.sum() / (masked_topk_p_x0 * conved_s[masked].unsqueeze(-1)).sum())  # scalar (variable of step). 
        conved_s_normed =  conved_s #* s_norm_constant   # [1, len]  
        confidence_conv =  confidence.unsqueeze(-1) * conved_s_normed.unsqueeze(-1) # [1, len, k] # normalizing term 은 j(candiate rank) agnostic하기때문에 unsqueeze 해줌.
        confidence_conv = confidence_conv.squeeze(-1)
        confidence = confidence_conv
        ##### -----------------------------------------
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]

    return x


def _subs_parameterization(logits, x, mask_id):
    logits[:, :, mask_id] += -99999
    # logits[:, :, 50256] += -99999  # 작은모델 '<|endoftext|>' .나중에 지우기.
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # log softmax
    unmasked_indices = (x != mask_id)
    logits[unmasked_indices] = -99999
    logits[unmasked_indices, x[unmasked_indices]] = 0
    return logits

def _sample_categorical_mdlm(categorical_probs, temperature ):  # gumbel softmax 로 categorical sampling 을 대신한것*****
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log()) ** temperature
    
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _sample_categorical_topk_gumbel_mdlm(categorical_probs , k, temperature=1):
    topk_val, topk_ind =categorical_probs.topk(k, dim=-1)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(topk_val) + 1e-10).log()) 
    gumbel_norm = gumbel_norm ** temperature
    argmax_in_topk = (topk_val / gumbel_norm).argmax(dim=-1)
    result = topk_ind.gather(dim=-1 , index=argmax_in_topk.unsqueeze(-1)).squeeze(-1)
    
    # return argmax_in_topk
    return result



def _ddpm_convolution_update(model,tokenizer, x, t, dt, mask_id, kernel, padding_size, k=1, conv_mult=1 ,temperature=1):
    k+=1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    
    logits = model(x).logits
    
    log_p_x0 = _subs_parameterization(logits, x, mask_id)
    p_x0 = log_p_x0.exp()
    topk_p_x0, topk_p_x0_ind = p_x0.topk(k= k, dim=-1)
    masked = x == mask_id
    
    #### convolutional decoding
    ### s_i
    unmasked = ~masked   # [batch, len]
    input_seq = unmasked.float().view(masked.size(0),1,-1).to('cuda')  # [batch, 1 , len]
    conv0 = F.conv1d(input_seq, kernel, stride=1, padding=padding_size.item() ).squeeze(1) # [batch,len] # max : window size. / when there is a left condition, condition end value: middle value of window size (e.g., 6 out of 11)
    normed_conv0 = conv0  # decided not to normalize. Can handle with conv_mult instead.
    conved_s = F.tanh(normed_conv0 * conv_mult) # normalization term s  [1, len]

    ## compute s_norm_constant.---------------------------------- 
    # denominator: at each position, sum_k(topk) * s, then sum only over mask positions
    eps0 = 1e-12
    den_per_pos = (topk_p_x0.sum(dim=-1) * conved_s)  # [B, L]

    den = (den_per_pos * masked.float()).sum(dim=-1)  # [B]

    # numerator: total sum of masked_p_x0 = number of masked tokens
    num = masked.sum(dim=-1).float()                  # [B]

    # normalization constant per batch
    s_norm_constant = (num / (den + eps0)).unsqueeze(-1)   # [B,1]  

    #-----------------------------------------------------------
    conved_s_normed =  conved_s * s_norm_constant   # [batch, len]  
    p_x0 =  topk_p_x0 * conved_s_normed.unsqueeze(-1) # [batch, len, k] # since the normalizing term is j(candiate rank) agnostic, unsqueeze
    # normalization value is based on masked positions, but spread across entire p_x0 -> no change in total prob. sum since copy_flag will overwrite later

    q_xs = p_x0 * (move_chance_t - move_chance_s)
    # q_xs.shape
    ### q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    q_xs[:, :, -1] = move_chance_s[:, :, 0]  # changed to top k, so shape is k+1. Last column is mask.
    # print(topk_p_x0_ind)
    topk_p_x0_ind[:,:,-1] = mask_id  # also change the index
    # print(topk_p_x0_ind)
    sampled_ind = _sample_categorical_mdlm(q_xs, temperature)
    _x = torch.gather(topk_p_x0_ind, -1, sampled_ind.unsqueeze(-1)).squeeze(-1)
    copy_flag = (x != mask_id).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x 


def _ddpm_update(model, tokenizer, x, t, dt, mask_id,  temperature=1):
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    
    logits = model(x).logits
    log_p_x0 = _subs_parameterization(logits, x, mask_id)
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)  
    q_xs[:, :, mask_id] = move_chance_s[:, :, 0]
    _x = _sample_categorical_mdlm(q_xs, temperature)
    
    copy_flag = (x != mask_id).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x   

    
def _ddpm_topk_update(model, tokenizer, x, t, dt, mask_id, k=1, conv_mult=1 ,temperature=1):
    k+=1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    logits = model(x).logits
    
    log_p_x0 = _subs_parameterization(logits, x, mask_id)
    p_x0 = log_p_x0.exp()
    
    assert move_chance_t.ndim == log_p_x0.ndim
    
    #####global norm 구하기  (whole norm -> global norm)
    masked = x == mask_id
    masked_p_x0 = p_x0[masked]#[:,:-1]    # [:,:-1] 왜들어갔는지 모르겠는데 계속 넣어서 하고있었음. 없어도 global_norm 값에 거의 영향 없음
    
    
    global_norm = masked_p_x0.sum() / masked_p_x0.topk(k=k, dim=-1).values.sum()

    q_xs = p_x0 * (move_chance_t - move_chance_s)  * global_norm

    q_xs[:, :, mask_id] = move_chance_s[:, :, 0]
    _x = _sample_categorical_topk_gumbel_mdlm(q_xs, k , temperature)
    
    copy_flag = (x != mask_id).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x    

def _remdm_topk_update(model, tokenizer, x, t, dt, mask_id, k=1, conv_mult=1 ,temperature=1, alpha_on=0.9, t_on=0.55, t_off=0.05, eta=0.04, last=False):
    k+=1
    # redmd-loop  in default
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    with torch.no_grad():
        logits = model(x).logits
    log_p_x0 = _subs_parameterization(logits, x, mask_id)
    p_x0 = log_p_x0.exp()
    
    if last:
        xs = p_x0.argmax(dim=-1)
        return xs
    
    time = t[0].item()
    # compute alpha_t and alpha_s
    if time > t_on:
        move_chance_t = (1 - (1 - t) * alpha_on / (1 - t_on))[:, None, None]
        move_chance_s = (1 - (1 - t + dt) * alpha_on / (1 - t_on))[:, None, None]
    elif time <= t_off:
        move_chance_t = (t * (1 - alpha_on) / t_off)[:, None, None]
        move_chance_s = ((t - dt) * (1 - alpha_on) / t_off)[:, None, None]
    else:
        move_chance_t, move_chance_s = None, None

    if time > t_on or time <= t_off:
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, mask_id] = move_chance_s[:, :, 0]
        _x = _sample_categorical_topk_gumbel_mdlm(q_xs, k, temperature)
        copy_flag = (x != mask_id).to(x.dtype)
        xs = copy_flag * x + (1 - copy_flag) * _x
    else: # use ReMDM
        sigma = eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., mask_id] = sigma
        q_xs_2 = p_x0 * ((alpha_on - (1 - sigma) * alpha_on) / (1 - alpha_on))
        q_xs_2[..., mask_id] = (1 - alpha_on - alpha_on * sigma) / (1 - alpha_on)
        copy_flag = (x != mask_id).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical_topk_gumbel_mdlm(q_xs, k, temperature)

    
    return xs



def _remdm_topk_conv_update(model, tokenizer, x, t, dt, mask_id, k=1, conv_mult=1, kernel=None, padding_size=None ,temperature=1, alpha_on=0.9, t_on=0.55, t_off=0.05, eta=0.04, last=False):
    k+=1
    # redmd-loop  in default
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    with torch.no_grad():
        logits = model(x).logits
    log_p_x0 = _subs_parameterization(logits, x, mask_id)
    p_x0 = log_p_x0.exp()
    
    if last:
        xs = p_x0.argmax(dim=-1)
        return xs
    
    
    #########################
    topk_p_x0, topk_p_x0_ind = p_x0.topk(k= k, dim=-1)
    masked = x == mask_id
    masked_topk_p_x0 = topk_p_x0[masked]
    masked_p_x0 = p_x0[masked]    #  sum^N sum^K  p(x_ij)

    ## convolutional decoding
    # s_i
    unmasked = ~masked   # [batch, len]
    input_seq = unmasked.float().view(masked.size(0),1,-1).to('cuda')  # [batch, 1 , len]
    conv0 = F.conv1d(input_seq, kernel, stride=1, padding=padding_size.item() ).squeeze(1) # [batch,len] # max : window size.   / 좌측 condition이 있을때, condition 끝 값: window size의 중간값. (e.g., 11 중 6)
    # normed_conv0 = conv0 / conv0.max()  # max=1 되게 norm. 무조건 1임. 
    normed_conv0 = conv0  # normalize 안하기로 함. conv_mult로 다 할 수 있음.
    conved_s = F.tanh(normed_conv0 * conv_mult) # normalization term s  [1, len]
    s_norm_constant = (masked_p_x0.sum() / (masked_topk_p_x0 * conved_s[masked].unsqueeze(-1)).sum())  # scalar (variable of step). 
    conved_s_normed =  conved_s * s_norm_constant   # [1, len]  
    p_x0 =  topk_p_x0 * conved_s_normed.unsqueeze(-1) 

    ##########################
    
    
    time = t[0].item()
    # compute alpha_t and alpha_s
    if time > t_on:
        move_chance_t = (1 - (1 - t) * alpha_on / (1 - t_on))[:, None, None]
        move_chance_s = (1 - (1 - t + dt) * alpha_on / (1 - t_on))[:, None, None]
    elif time <= t_off:
        move_chance_t = (t * (1 - alpha_on) / t_off)[:, None, None]
        move_chance_s = ((t - dt) * (1 - alpha_on) / t_off)[:, None, None]
    else:
        move_chance_t, move_chance_s = None, None

    if time > t_on or time <= t_off:
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, -1] = move_chance_s[:, :, 0]
        topk_p_x0_ind[:,:,-1] = mask_id
        sampled_ind = _sample_categorical_mdlm(q_xs, temperature)
        _x = torch.gather(topk_p_x0_ind, -1, sampled_ind.unsqueeze(-1)).squeeze(-1)
        copy_flag = (x != mask_id).to(x.dtype)
        xs = copy_flag * x + (1 - copy_flag) * _x
    else: # use ReMDM
        sigma = eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., -1] = sigma
        q_xs_2 = p_x0 * ((alpha_on - (1 - sigma) * alpha_on) / (1 - alpha_on))
        q_xs_2[..., -1] = (1 - alpha_on - alpha_on * sigma) / (1 - alpha_on)
        topk_p_x0_ind[..., -1] = mask_id
        copy_flag = (x != mask_id).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        sampled_ind = _sample_categorical_mdlm(q_xs,  temperature)
        xs = torch.gather(topk_p_x0_ind, -1, sampled_ind.unsqueeze(-1)).squeeze(-1)
    
    return xs

  

def _ddpm_topk_penalty_update(model, tokenizer, x, t, dt, mask_id, k=1, conv_mult=1 ,temperature=1, repetition_penalty=0.2):
    k+=1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    logits = model(x).logits

    penalty = float(repetition_penalty)
    x_expanded = x.unsqueeze(1).expand(-1, x.size(-1), -1)
    score = torch.gather(logits, dim=2, index=x_expanded.to(torch.int64))
    # score = torch.where(score < 0, score * penalty ,  score / penalty)   # dtype = penalty
    score /= penalty  # dtype = penaltyctrl
    logits = logits.scatter(-1, x_expanded.to(torch.int64), score)

    
    log_p_x0 = _subs_parameterization(logits, x, mask_id)
    p_x0 = log_p_x0.exp()
    
    assert move_chance_t.ndim == log_p_x0.ndim
    
    #####global norm 구하기  (whole norm -> global norm)
    masked = x == mask_id
    masked_p_x0 = p_x0[masked]#[:,:-1]    # [:,:-1] 왜들어갔는지 모르겠는데 계속 넣어서 하고있었음. 없어도 global_norm 값에 거의 영향 없음
    
    
    global_norm = masked_p_x0.sum() / masked_p_x0.topk(k=k, dim=-1).values.sum()

    q_xs = p_x0 * (move_chance_t - move_chance_s)  * global_norm

    q_xs[:, :, mask_id] = move_chance_s[:, :, 0]
    _x = _sample_categorical_topk_gumbel_mdlm(q_xs, k , temperature)
    
    copy_flag = (x != mask_id).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x    

    
@ torch.no_grad()
def generate_yb(model,tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0., mask_id=126336, k=1, conv_mult=1,eos_fill=False, decode_type='', semi_ar=False, repetition_penalty=0.2, alpha_on=0.9, t_on=0.55, t_off=0.05, eta=0.04, bidirection=False, bi_what='eos', prompt_original=False): 
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    if semi_ar: # inside semi_ar
        x = prompt
    else:
        if prompt_original:
            x = prompt.clone()
        else:
            x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, :prompt.shape[1]] = prompt.clone()
    
    if bidirection:
        if bi_what in ['eos', '', None]:
            x[...,-1] = tokenizer.eos_token_id
        else:
            add = tokenizer.encode(bi_what, return_tensors='pt')[0]
            x[..., - (len(add) + 20 ): -20] = add
    
    eps=1e-5
    timesteps = torch.linspace( 1, eps, steps + 1, device=model.device)  # 1 -> 0  을 num_step +1 으로 나눔.
    dt = (1 - eps) / steps
    kernel_size = block_length  ########
    if kernel_size % 2 !=1: 
        kernel_size +=1
    kernel = torch.ones(kernel_size).to(model.device)
    kernel = kernel.view(1, 1, -1)   # [1,1, window_size]
    padding_size = torch.tensor((kernel_size -1 ) // 2 ).to(model.device)
    
    prompt_index = (x != mask_id)
    # kernel size = block length 
    
    for i in range(steps):
        # print(i)
        t = timesteps[i] * torch.ones(x.shape[0], device= model.device) # shape 는 (batch size, 1)
        if decode_type=='conv':
            x = _ddpm_convolution_update(model,tokenizer, x,t, dt, mask_id, kernel, padding_size, k, conv_mult, temperature=1)
            # print(x)
            
        elif decode_type=='topk':
            x = _ddpm_topk_update(model,tokenizer, x, t, dt, mask_id, k, conv_mult ,temperature=1)
            # print(x)
        elif decode_type=='topk_penalty':
            x = _ddpm_topk_penalty_update(model,tokenizer, x, t, dt, mask_id, k, conv_mult ,temperature=1, repetition_penalty=repetition_penalty)
            # print(x)
        elif decode_type=='categorical':
            x= _ddpm_update(model,tokenizer, x, t, dt, mask_id, temperature=1)
            
        elif decode_type=='remdm_loop_topk':
            x = _remdm_topk_update(model, tokenizer, x, t, dt, mask_id, k=k, conv_mult=1 ,temperature=1, alpha_on=alpha_on, t_on=t_on, t_off=t_off, eta=eta, last= i==steps-1)
        elif decode_type=='remdm_loop_conv':
            x = _remdm_topk_conv_update(model, tokenizer, x, t, dt, mask_id, k=k, conv_mult=1, kernel=kernel , padding_size=padding_size, temperature=1, alpha_on=alpha_on, t_on=t_on, t_off=t_off, eta=eta, last= i==steps-1)
        else:
            raise Exception("Unknown decode type !!!!!!!!!!!!!")
            
        if eos_fill:  
            eos_found = ((x == tokenizer.eos_token_id).cumsum(dim=-1) >=2)
            x[eos_found] = tokenizer.eos_token_id        
        
    return x


@torch.no_grad()
def generate_slide_annealing_update(model,tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0., mask_id=126336, k=1, conv_mult=1,eos_fill=False, decode_type='', semi_ar=False, repetition_penalty=0.2, alpha_on=0.9, t_on=0.55, t_off=0.05, eta=0.04, bidirection=False, bi_what='eos'):
    k+=1
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    eps=1e-5
    timesteps = torch.linspace( 1, eps, steps + 1, device=model.device)  # 1 -> 0  을 num_step +1 으로 나눔.
    dt = (1 - eps) / steps
    r = int(gen_length / steps)
    
    for i in range(steps):
        
        
        t = timesteps[i] * torch.ones(x.shape[0],  device= model.device)

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        logits = model(x).logits
        log_p_x0 = _subs_parameterization(logits, x, mask_id)
        p_x0 = log_p_x0.exp()
            
        
        for j in range(p_x0.shape[0]):
            start_ind = prompt_index[j].sum(dim=-1) #- 1  ##
            
            ### 1) front
            front_p = p_x0[j, start_ind + r * i : start_ind + r * (i + 1) ]
            # print(f"{start_ind + r * i}     {start_ind + r * (i + 1)} ")
            topk_val, topk_ind =front_p.topk(k, dim=-1)
            gumbel_norm = (
                1e-10
                - (torch.rand_like(topk_val) + 1e-10).log()) 
            gumbel_norm = gumbel_norm ** temperature
            
            argmax_in_topk = (topk_val / gumbel_norm).argmax(dim=-1)
            result_front = topk_ind.gather(dim=-1 , index=argmax_in_topk.unsqueeze(-1)).squeeze(-1)
            
            ### 2) reer
            reer_p = p_x0[j,  start_ind + r * (i + 1) : start_ind + r * i + block_length]
            global_norm = reer_p.sum() / reer_p.topk(k=k, dim=-1).values.sum()
            q_xs = reer_p * ((move_chance_t - move_chance_s).squeeze())  * global_norm
            # q_xs = reer_p * global_norm
            # q_xs = reer_p 
            # q_xs[:,  mask_id] = 0.2 # move_chance_s[:, :, 0]
            q_xs[:,  mask_id] = move_chance_s[:, :, 0].squeeze(0)[0]
            
            topk_val, topk_ind =q_xs.topk(k, dim=-1)
            gumbel_norm = (
                1e-10
                - (torch.rand_like(topk_val) + 1e-10).log()) 
            gumbel_norm = gumbel_norm ** temperature
            
            argmax_in_topk = (topk_val / gumbel_norm).argmax(dim=-1)
            result_reer = topk_ind.gather(dim=-1 , index=argmax_in_topk.unsqueeze(-1)).squeeze(-1)
            
            
            x[j, start_ind + r * i : start_ind + r *i + r] = result_front
            x[j, start_ind + r*i + r : start_ind + r * i + block_length] = result_reer 
    return x

# @ torch.no_grad()
# def 


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True,cache_dir="/convei_nas2/huggingface_cache/hub" , torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct',cache_dir="/convei_nas2/huggingface_cache/hub" , trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_yb(model, tokenizer, input_ids, steps=8, gen_length=512, block_length=128, temperature=1., mask_id=126336, k=1,eos_fill=False, decode_type='topk')
    # print(out)
    
    print(tokenizer.batch_decode(out)[0])


if __name__ == '__main__':
    main()
