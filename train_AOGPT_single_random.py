"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import data
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from ema import ExponentialMovingAverage

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
save_interval = 10000
log_interval = 1
eval_iters = 200
eval_monte_carlo_steps = 10
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
train_dataset = 'openwebtext'
valid_dataset = 'wikitext103'
val_dataset_list = ['wikitext2', 'wikitext103', 'ptb', 'lambada']
cache_dir = '/home/cache'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
model_type = None
# EMA
ema_rate = 1-1e-4
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
order = None
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


if model_type == 'AdaLN6_NoRep_cond_128_trunc_qknorm':
    from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig
else:
    raise NotImplementedError(f"Model Type {model_type} is not supported")

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    assert eval_iters % ddp_world_size == 0
    eval_iters //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

order = torch.tensor(order)
orders = torch.tile(order, (batch_size, 1)).to(device)


# Torch Train and Val Dataloader
train_loader, val_loader = data.get_dataloaders(train_dataset=train_dataset, valid_dataset=valid_dataset, batch_size=batch_size, cache_dir=cache_dir, block_size=block_size, distributed=ddp, n_proc=64)
train_iter, val_iter = iter(train_loader), iter(val_loader)
def get_batch(split):
    if split == 'train':
        x = next(train_iter)['input_ids'].to(device)
    else:
        x = next(val_iter)['input_ids'].to(device)

    return x

# Additional Val Dataloader for Zero-shot PPL
# Get wikitext2
wikitext2_dataloader = data.get_valid_dataloaders(dataset_name='wikitext2', batch_size=batch_size, cache_dir=cache_dir, block_size=block_size, distributed=ddp)
# Get wikitext103
wikitext103_dataloader = data.get_valid_dataloaders(dataset_name='wikitext103', batch_size=batch_size, cache_dir=cache_dir, block_size=block_size, distributed=ddp)   
# Get ptb
ptb_dataloader = data.get_valid_dataloaders(dataset_name='ptb', batch_size=batch_size, cache_dir=cache_dir, block_size=block_size, distributed=ddp)   
# Get lambada
lambada_dataloader = data.get_valid_dataloaders(dataset_name='lambada', batch_size=batch_size, cache_dir=cache_dir, block_size=block_size, distributed=ddp)   
# Get lm1b
lm1b_dataloader = data.get_valid_dataloaders(dataset_name='lm1b', batch_size=batch_size, cache_dir=cache_dir, block_size=block_size, distributed=ddp) 
dataloader_dic = {
    'wikitext2': wikitext2_dataloader,
    'wikitext103': wikitext103_dataloader,
    'ptb': ptb_dataloader,
    'lambada': lambada_dataloader,
    'lm1b': lm1b_dataloader
}
AR_ppl_dic = {
    'wikitext2': 1e8,
    'wikitext103': 1e8,
    'ptb': 1e8,
    'lambada': 1e8,
    'lm1b': 1e8
}
Random_ppl_dic = {
    'wikitext2': 1e8,
    'wikitext103': 1e8,
    'ptb': 1e8,
    'lambada': 1e8,
    'lm1b': 1e8
}

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = 50304
    AOGPTconf = AOGPTConfig(**model_args)
    model = AOGPT(AOGPTconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    AOGPTconf = AOGPTConfig(**model_args)
    model = AOGPT(AOGPTconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    
# create ema copy of model
ema = ExponentialMovingAverage(
    model.parameters(), decay=ema_rate)
print(f"EMA: {ema_rate}")

# helps estimate an arbitrarily accurate loss over either split using many batches
def estimate_loss_AR():
    out = {}
    for split in ['train', 'val']:
        total_loss = 0
        for k in range(eval_iters):
            X = get_batch(split)
            with ctx:
                _, loss = model(X, orders=orders)
            total_loss += loss
        dist.all_reduce(total_loss)
        total_loss /= eval_iters
        total_loss /= ddp_world_size
        out[split] = total_loss.item()
    return out

def estimate_loss_Random():
    out = {}
    for split in ['train', 'val']:
        total_loss = 0
        for k in range(eval_iters):
            X = get_batch(split)
            loss = 0
            with ctx:
                for _ in range(eval_monte_carlo_steps):
                    _, per_mc_step_loss = model(X, mode='Random')
                    loss += per_mc_step_loss
            loss /= eval_monte_carlo_steps
            total_loss += loss
        dist.all_reduce(total_loss)
        total_loss /= eval_iters
        total_loss /= ddp_world_size
        out[split] = total_loss.item()
    return out

def estimate_PPL_AR():
    for dataset_name in val_dataset_list:
        dataloader = dataloader_dic[dataset_name]
        ppl_eval_iter = iter(dataloader)
        total_loss = 0
        batch_num = 0
        for batch in ppl_eval_iter:
            batch = batch['input_ids'].to(device)
            _, cur_loss = model(batch, orders=orders)
            total_loss += cur_loss
            batch_num += 1
        dist.all_reduce(total_loss)
        total_loss /= ddp_world_size
        total_loss /= batch_num
        ppl = math.exp(total_loss)
        AR_ppl_dic[dataset_name] = ppl
        if master_process:
            print(f'Dataset: {dataset_name}')
            print(f"Evaluation PPL: {ppl}")

def estimate_PPL_Random():
    for dataset_name in val_dataset_list:
        dataloader = dataloader_dic[dataset_name]
        ppl_eval_iter = iter(dataloader)
        total_loss = 0
        batch_num = 0
        for batch in ppl_eval_iter:
            batch = batch['input_ids'].to(device)
            cur_loss = 0
            for _ in range(eval_monte_carlo_steps):
                _, per_mc_step_cur_loss = model(batch, mode='Random')
                cur_loss += per_mc_step_cur_loss
            cur_loss /= eval_monte_carlo_steps
            total_loss += cur_loss
            batch_num += 1
        dist.all_reduce(total_loss)
        total_loss /= ddp_world_size
        total_loss /= batch_num
        ppl = math.exp(total_loss)
        Random_ppl_dic[dataset_name] = ppl
        if master_process:
            print(f'Dataset: {dataset_name}')
            print(f"Evaluation PPL: {ppl}")


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num > 0 and iter_num % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            AR_losses = estimate_loss_AR()
            if master_process:
                print(f"AR Likelihood: step {iter_num}: train loss {AR_losses['train']:.4f}, val loss {AR_losses['val']:.4f}")
            Random_losses = estimate_loss_Random()
            if master_process:
                print(f"Random ELBO: step {iter_num}: train loss {Random_losses['train']:.4f}, val loss {Random_losses['val']:.4f}")
                print('Start Evaluate AR PPL')
            # Estimate AR PPL
            estimate_PPL_AR()
            if master_process:
                print('Start Evaluate Random PPL')
            # Estimate Random PPL    
            estimate_PPL_Random()
            ema.restore(model.parameters())
        model.train()
        if wandb_log and master_process:
            wandb_log_dic = {
                "iter": iter_num,
                "train/AR_loss": AR_losses['train'],
                "val/AR_loss": AR_losses['val'],
                "train/Random_loss": Random_losses['train'],
                "val/Random_loss": Random_losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            AR_val_ppl_dic = {f'{k}_AR': AR_ppl_dic[k] for k in val_dataset_list}
            Random_val_ppl_dic = {f'{k}_Random': Random_ppl_dic[k] for k in val_dataset_list}
            wandb.log({**wandb_log_dic, **AR_val_ppl_dic, **Random_val_ppl_dic})

        if iter_num > 0 and iter_num % save_interval == 0 and master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'ema_model': ema.state_dict(),
                'ema_rate': ema_rate,
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, orders=orders)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    ema.update(model.parameters())
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    dist.barrier()
    destroy_process_group()
