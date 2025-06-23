import torch
import argparse

from model_AOGPT_generate import AOGPTConfig, AOGPT
import torch.nn.functional as F
import os
import numpy as np
import time



device = torch.device('cuda')
ckpt_dir = 'ckpt_dir_here' # e.g.  '/home/ckpt/ckpt_500000.pt'
checkpoint = torch.load(ckpt_dir, map_location=device)
gptconf = AOGPTConfig(**checkpoint['model_args'])
model = AOGPT(gptconf)
ema_params = checkpoint['ema_model']['shadow_params']
for s_param, param in zip(ema_params, model.parameters()):
    param.data.copy_(s_param.data)
model.to(device)
model.eval()


batch_size = 8
sampling_orders = torch.stack([torch.randperm(1024, device=device) for _ in range(batch_size)])

num_warmup=10
num_repeats=100

# warmup
print(f"{num_warmup} warmups")
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _ in range(num_warmup):
            _ = model.forward_fn_generate_first_step(sampling_orders)


torch.cuda.synchronize()



print(f"{num_repeats} measurements")
timings = []
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _ in range(num_repeats):
            start_time = time.time()
            _ = model.forward_fn_generate_first_step(sampling_orders)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)

mean_time = np.mean(timings) * 1000  # to ms
std_time = np.std(timings) * 1000
min_time = np.min(timings) * 1000
max_time = np.max(timings) * 1000

print(f"\nStatistics")
print(f"Average Time: {mean_time:.2f} ms")
print(f"Std: {std_time:.2f} ms")
print(f"Min Time: {min_time:.2f} ms")
print(f"Max Time: {max_time:.2f} ms")