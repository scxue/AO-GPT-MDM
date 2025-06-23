# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Model size
n_layer = 12
n_head = 12
n_embd = 768
model_type = 'AdaLN6_NoRep_cond_128_trunc_qknorm'

# I/O
out_dir = 'v3_ckpt_NoCL/124M-AdaLN6_NoRep_cond_128_trunc_bs1M_wd_0.03_drop_0.0_CL_0.0_qknorm'
wandb_log = True
wandb_project = 'AOGPT'
wandb_run_name= 'AOGPT-124M-AdaLN6_NoRep_cond_128_trunc_bs1M_wd_0.03_drop_0.0_CL_0.0_qknorm'

# Bias
bias = True

# Dropout
dropout = 0.0

# EMA
ema_rate = 1 - 1e-4

# weight decay
weight_decay = 3e-2

# beta2
beta2 = 0.95

# these make the total batch size be ~0.5M
# 16 batch size * 1024 block size * 8 gradaccum * 8 GPUs = 524,288 * 2
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 8*8

# this makes total number of tokens be 500B
learning_rate = 6e-4
max_iters = 500000
lr_decay_iters = 500000

# eval stuff
eval_interval = 2500
save_interval = 10000
eval_iters = 96
log_interval = 10

# MDM setting
random_ratio = 1.0


