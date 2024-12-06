# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Model size
n_layer = 24
n_head = 16
n_embd = 1024

# I/O
out_dir = '350M'
wandb_log = True
wandb_project = 'CADD'
wandb_run_name= 'CADD-350M'

# these make the total batch size be ~0.5M
# 16 batch size * 1024 block size * 4 gradaccum * 8 GPUs = 524,288
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 4*8

# this makes total number of tokens be 500B
learning_rate = 3e-4
max_iters = 1000000
lr_decay_iters = 1000000

# eval stuff
eval_interval = 5000
save_interval = 10000
eval_iters = 96
log_interval = 10

# weight decay
weight_decay = 1e-1
