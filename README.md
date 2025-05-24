# Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning

---

### Abstract

Autoregressive (AR) language models generate text one token at a time, which limits their inference speed. Diffusion-based language models offer a promising alternative, as they can decode multiple tokens in parallel. However, we identify a key bottleneck in current diffusion LMs: the **long decoding-window problem**, where tokens generated far from the input context often become irrelevant or repetitive. Previous solutions like semi-autoregressive address this issue by splitting windows into blocks, but this sacrifices speed and bidirectionality, eliminating the main advantage of diffusion models. To overcome this, we propose **Convolutional decoding (*Conv*)**, a normalization-based method that narrows the decoding window without hard segmentation, preserving fluency and bidirectionality. Additionally, we introduce **Rejecting Rule-based Fine-Tuning (R2FT)**, a post-hoc training scheme that better aligns tokens at positions far from context. Our methods achieve state-of-the-art results on open-ended generation benchmarks (e.g., AlpacaEval) among diffusion LM baselines, with significantly lower step size than previous works, demonstrating both speed and quality improvements. 

---

### Installation

```python
CUDA 12.1
$ conda create --name conv python=3.9
$ conda activate conv
$ pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

$ pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3.post1/causal_conv1d-1.1.3.post1+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

$ pip install datasets==2.18.0 einops==0.7.0 fsspec==2024.2.0 git-lfs==1.6 h5py==3.10.0 hydra-core==1.3.2 ipdb==0.13.13 lightning==2.2.1 

$ pip install https://github.com/state-spaces/mamba/releases/download/v1.1.4/mamba_ssm-1.1.4+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

$ pip install notebook==7.1.1 nvitop==1.3.2 omegaconf==2.3.0 packaging==23.2 pandas==2.2.1 rich==13.7.1 seaborn==0.13.2 scikit-learn==1.4.0 timm==0.9.16 transformers==4.38.2 triton==2.2.0 wandb==0.13.5 

$ pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

$ pip install bitsandbytes==0.42.0 git+https://github.com/huggingface/peft.git 
```

---

### Quick generation with convolutional decoding

Generating responses with already fine-tuned model from [2] (LLaDA-8B-Instruct).

```
bash scripts/generate_large_inst.sh
```

This bash script contains the following command, which generate with ***Conv* (convolutional decoding).**

```
python gen1_1_answer_generation.py \
init_from_checkpoint.init_file='' \
data.tokenizer_name_or_path=GSAI-ML/LLaDA-8B-Instruct \
backbone=llada_inst \
lora.bool=True \
parameterization=subs \
sampling.steps=64 \
model.length=512 \
model_max_length=512 \
sampling.topk_k=1 \
sampling.convolution_kernel_size=128\
sampling.predictor=ddpm_convolution \
+category=gsm8k_large \
+eval_data=gsm8k \
+generator=llada_inst_ddpm_conv_L512_S64 \
master_port=65535 \
+rand_value=111 \
+batch_size=8
```

---

## Convolutional decoding

![conv_pipeline.png](attachment:9f431e32-f638-48e4-aa3b-6d507c025e5c:conv_pipeline.png)

Implementation for convolutional decoding is in    `decoding_tools.py >  def ddpm*_*convolution_update()`

```python
...
log_p_x0 = self.forward(x, unet_conditioning)
p_x0 = log_p_x0.exp()
topk_p_x0, topk_p_x0_ind = p_x0.topk(k= k, dim=-1)
masked = x == self.mask_index
masked_topk_p_x0 = topk_p_x0[masked]
masked_p_x0 = p_x0[masked]
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
...
```

---

# Training

We provide code for SFT and RFT

### Tokenize datasets

First, the dataset is tokenized and saved.

```python
python tr1_make_dataset.py \
+model_type=small

python tr1_make_dataset.py \
+model_type=large

python tr1_make_dataset.py \
+model_type=llama
```

### Download checkpoints

For SFT and RFT, download the pretrained checkpoints as follows: the small model from [1], and the large model from [2].

### SFT

Commands for SFT.

```bash
bash scripts/sft_small.sh
bash scripts/sft_large.sh
```

This script contains the following command.

```
python tr2_finetune.py \
data.tokenizer_name_or_path=gpt2 \
lora.bool=False \
finetune.bool=True \
backbone=dit \
wandb.name=ft_small \
parameterization=subs \
T=0 \
model.length=1024 \
eval.retokenize=True \
save_weight_only=True \
forward_type=ddpm \
optim.lr=2.5e-5 \
optim.weight_decay=0.1 \
loader.global_batch_size=256 \
lr_scheduler.num_warmup_steps=500 \
sampling.predictor=ddpm \
sampling.num_sample_log=6 \
loader.batch_size=1 \
loader.eval_batch_size=1 \
finetune.attention_cover=response \
finetune.dataset=src_data/ft_data/tokenized_alpaca_instruction \
val_eos_off=False
```

### R2FT

Commands for R2FT with SFT checkpoints.

```bash
bash scripts/r2ft_small.sh
bash scripts/r2ft_large.sh
```

This script contains the following command.

```
python tr2_finetune.py \
init_from_checkpoint.bool=True \
init_from_checkpoint.init_file=outputs/ft_small.ckpt \
data.tokenizer_name_or_path=gpt2 \
lora.bool=False \
finetune.bool=True \
r2ft.bool=True \
r2ft.gamma=0 \
r2ft.beta_w=1 \
r2ft.beta_l=1 \
r2ft.beta_a=0.1 \
backbone=dit \
wandb.name=r2ft_small \
parameterization=subs \
T=0 \
model.length=1024 \
eval.retokenize=True \
save_weight_only=True \
forward_type=ddpm \
optim.lr=2.5e-5 \
optim.weight_decay=0.1 \
loader.global_batch_size=256 \
lr_scheduler.num_warmup_steps=500 \
sampling.predictor=ddpm \
sampling.num_sample_log=6 \
loader.batch_size=1 \
loader.eval_batch_size=1 \
finetune.attention_cover=response \
finetune.dataset=src_data/ft_data/tokenized_alpaca_instruction \
finetune.valid_size=20 \
val_eos_off=False
```

### Generate

```bash
bash scripts/generate_small.sh
bash scripts/generate_large.sh
```

---

## Evaluation

The evaluation for AlpacaEval uses the code from the **[3].**

The evaluation for GSM8K is as following.

```bash
python eval_gsm8k.py \
--eval_data=llada_inst_ddpm_conv_L1024_S128   # << subdir name in answer_generation/generated >>
```

---

### Copyright

These codes are based on the GitHub repository of MDLM [1], and partially LLADA[2].

---

## Reference

[1] Sahoo, Subham, et al. "Simple and effective masked diffusion language models." *Advances in Neural Information Processing Systems* 37 (2024)

[2] Nie, Shen, et al. "Large language diffusion models." *arXiv preprint arXiv:2502.09992* (2025).

[3]Dubois, Yann, et al. "Length-controlled alpacaeval: A simple way to debias automatic evaluators." *arXiv preprint arXiv:2404.04475* (2024).