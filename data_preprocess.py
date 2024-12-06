import data
import torch.distributed as dist
dist.init_process_group()
train_dataset = 'openwebtext'
valid_dataset = 'wikitext103'
cache_dir = '/root/autodl-tmp/cache'
batch_size = 12
block_size = 1024


train_loader, val_loader = data.get_dataloaders(train_dataset=train_dataset, valid_dataset=valid_dataset, batch_size=batch_size, cache_dir=cache_dir, block_size=block_size, distributed=True, n_proc=128)