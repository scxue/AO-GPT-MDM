# import numpy as np
# from transformers import GPT2TokenizerFast
# data = np.memmap('/root/autodl-tmp/data/openwebtext/train.bin', dtype=np.uint16, mode='r')
# tokenizer = GPT2TokenizerFast.from_pretrained('/root/autodl-tmp/model/gpt2')
# print(tokenizer.decode(data[0:2048]))
# print(data[0:1024])
# dataset_name = 'a'
# ppl = 1
# print("================================")
# print(f'Dataset: {dataset_name}')
# print(f"Evaluation PPL: {ppl}")
# print("================================")

# print("================================")
# print(f'Dataset: {dataset_name}')
# print(f"Evaluation PPL: {ppl}")
# print("================================")

# a = {
#                 "iter": 0,
#                 "train/loss": 0,
#                 "lr": 0,
#                 "mfu": 0, # convert to percentage
#             }

# b = ppl_dic = {
#     'wikitext2': 1e8,
#     'wikitext103': 1e8,
#     'ptb': 1e8,
#     'lambada': 1e8,
#     'lm1b': 1e8
# }
# val_dataset_list = ['wikitext2', 'wikitext103', 'ptb', 'lambada']
# val_ppl_dic = {k: ppl_dic[k] for k in val_dataset_list}
# print(val_ppl_dic)

# from datasets import load_dataset
# dataset = load_dataset("/root/autodl-tmp/data/openwebtext", cache_dir='/root/autodl-tmp/cache', num_proc=16)
# for i in range(30):
#     print(len(dataset["train"][i]['text']))
#     tokens = tokenizer(dataset["train"][i]['text'])
#     print(len(tokens['input_ids']))
    
# print(tokenizer)

# import torch

# batch = 2
# seq_len = 3

# x = torch.randint(0,20,[batch, seq_len])
# print('x:', x)

# shuffled_orders = []
# for _ in range(batch):
#     shuffled_orders.append(torch.randperm(seq_len, device=x.device))
# shuffled_orders = torch.stack(shuffled_orders)

# print('shuffled_orders:', shuffled_orders)

# batch_indices = torch.arange(batch).unsqueeze(1).expand(-1, seq_len)
# shuffled_x = x[batch_indices, shuffled_orders]

# print('shuffled_x:', shuffled_x)


# batch_indices = torch.arange(batch).unsqueeze(1).expand(-1, seq_len)
# unshuffled_x = torch.zeros_like(shuffled_x)
# unshuffled_x[batch_indices, shuffled_orders] = shuffled_x

# print('unshuffled_x:', unshuffled_x)

# import torch.nn as nn

# idx = torch.tensor([[1,3,5], [2,4,6]])
# emb = nn.Embedding(1, 4096)
# print(idx[:,1:].shape)
# print(idx.shape[0])


# noneemb = emb(torch.tensor([[0]]))
# print(noneemb.shape)
# print(torch.tensor([[0]]).shape)



# a = torch.tensor([[1,3,5], [2,4,6]])

# print(a.shape)
# print(emb(a).shape)

# def shuffle(x, orders):
#     batch_size, seq_len = x.shape[:2]
#     batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
#     shuffled_x = x[batch_indices, orders]
#     return shuffled_x

# # From rar https://github.com/bytedance/1d-tokenizer/blob/main/modeling/rar.py#L295
# def unshuffle(shuffled_x, orders):
#     # Unshuffle the tensor based on the original orders
#     batch_size, seq_len = shuffled_x.shape[:2]
#     batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
#     unshuffled_x = torch.zeros_like(shuffled_x)
#     unshuffled_x[batch_indices, orders] = shuffled_x
#     return unshuffled_x

# batch = 2
# seq_len = 3

# shuffled_orders = []
# for _ in range(batch):
#     shuffled_orders.append(torch.randperm(seq_len))
# shuffled_orders = torch.stack(shuffled_orders)

# print(shuffled_orders)

# print(emb(shuffle(a, shuffled_orders)))
# print(shuffle(emb(a), shuffled_orders))

# print(emb(shuffle(a, shuffled_orders)) == shuffle(emb(a), shuffled_orders))

# print(emb(a)[:,0].unsqueeze(1).shape)

# print(a)
# print(emb(a)[:,0])

# import data
# import torch.distributed as dist
# from transformers import GPT2TokenizerFast
# dist.init_process_group()
# train_dataset = 'openwebtext'
# valid_dataset = 'wikitext103'
# cache_dir = '/root/autodl-tmp/cache'
# batch_size = 12
# block_size = 1024

# tokenizer = GPT2TokenizerFast.from_pretrained('/root/autodl-tmp/model/gpt2')

# enc = tokenizer('why the begin of sentence issue is so messy and problematic')['input_ids']

# print(enc)
# print(tokenizer.decode(enc))
# train_loader, val_loader = data.get_dataloaders(train_dataset=train_dataset, valid_dataset=valid_dataset, batch_size=batch_size, cache_dir=cache_dir, block_size=block_size, distributed=True, n_proc=64)

# train_iter= iter(train_loader)
# device = 'cuda'

# def get_batch(split):
#     if split == 'train':
#         x = next(train_iter)['input_ids'].to(device)

#     return x
# x = get_batch('train')
# for i in range(12):
#     print(x[i])
# x = get_batch('train')
# for i in range(12):
#     print(x[i])
# x = get_batch('train')
# for i in range(12):
#     print(x[i])
# print(tokenizer)
# print(tokenizer.batch_decode(x)[0])
# print(tokenizer.batch_decode(x)[1])


# dist.destroy_process_group()
import torch
from model_CADD import CADDConfig, CADD

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
block_size = 1024
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
model_args['vocab_size'] = 50304
gptconf = CADDConfig(**model_args)
model = CADD(gptconf)

idx = torch.randint(0, 4096*2, [16,1024])
print(idx.shape)
print(model(idx))