# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Model size
n_layer = 12
n_head = 12
n_embd = 768
model_type = 'AdaLN6_NoRep_cond_128_trunc_qknorm'

# I/O
out_dir = 'order_ckpt/124M-AR'
wandb_log = True
wandb_project = 'AOGPT'
wandb_run_name= 'AOGPT-124M-AR'

# Bias
bias = True

# Dropout
dropout = 0.0

# EMA
ema_rate = 1 - 1e-4

# weight decay
weight_decay = 5e-2

# beta2
beta2 = 0.95

# these make the total batch size be ~0.5M
# 32 batch size * 1024 block size * 2 gradaccum * 8 GPUs = 524,288
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 2*8

# this makes total number of tokens be 500B
learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 500000

# eval stuff
eval_interval = 1000
save_interval = 10000
eval_iters = 96
log_interval = 10


