import data
import torch.distributed as dist
dist.init_process_group()
train_dataset = 'openwebtext'
valid_dataset = 'wikitext103'
cache_dir = '/home/cache' # your cache for data here
batch_size = 16
block_size = 1024


train_loader, val_loader = data.get_dataloaders(train_dataset=train_dataset, valid_dataset=valid_dataset, batch_size=batch_size, cache_dir=cache_dir, block_size=block_size, distributed=True, n_proc=64)
ddp = True
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