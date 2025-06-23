"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import torch.distributed as dist
from model_AOGPT_generate import AOGPTConfig, AOGPT
from transformers import GPT2TokenizerFast
import time
import numpy as np
import torch.nn.functional as F

# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
batch_size = 32
sampling_steps = 512
seed = 4234
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


device = 'cuda:2'

master_process = True
seed_offset = 0
ddp_world_size = 1

torch.manual_seed(seed + seed_offset)
torch.cuda.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# model
# init from a model saved in a specific directory
ckpt_dir = '/home/ckpt/350M/350M-AdaLN6_NoRep_cond_128_trunc_bs1M_wd_0.03_drop_0.0_CL_0.1_qknorm/ckpt_460000.pt'
checkpoint = torch.load(ckpt_dir, map_location=device)
gptconf = AOGPTConfig(**checkpoint['model_args'])
model = AOGPT(gptconf)
ema_params = checkpoint['ema_model']['shadow_params']
for s_param, param in zip(ema_params, model.parameters()):
    param.data.copy_(s_param.data)
model.to(device)
model.eval()




# sampling_orders = torch.stack([torch.arange(model.config.block_size, device=device) for _ in range(num_samples)])
start_time = time.time()
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        idx = model.generate(batch_size, sampling_steps, device, schedules='linear', topp=1, temperature=1)
print('time:', time.time()-start_time)
# print(tokenizer.batch_decode(idx[0:3]))

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
eval_model = GPT2LMHeadModel.from_pretrained('openai/gpt2-large').to(device).eval()
tokenizer = GPT2TokenizerFast.from_pretrained('openai/gpt2-large')

total_perplexity = []
with torch.no_grad():
    batches = idx.shape[0] // 16
    for i in range(batches):
        s = idx[i * 16:(i + 1) * 16]
        _, logits = eval_model(s, labels=s)[:2]
        logits = logits.transpose(-1, -2)
        perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp()
        total_perplexity.extend(list(perplexity.cpu().numpy()))
print(f"Generative Perplexity Mean: {np.mean(total_perplexity):.3f}.\n")
print(f"Generative Perplexity Std: {np.std(total_perplexity):.3f}.\n")