import torch
import data
import argparse
import os
import torch.distributed as dist
import math
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from torch.distributed import init_process_group, destroy_process_group



def main():
    parser = argparse.ArgumentParser(description="Evaluation Perplexity")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cache_dir", type=str, default="/home/cache")
    parser.add_argument("--block_size", type=int, default=1024)
    
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    
    model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2-medium').to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2-medium')
    args = parser.parse_args()
    args.distributed = (ddp_world_size > 1)
    args.ngpus = ddp_world_size
    # Get openwebtext
    train_loader, _ = data.get_dataloaders(train_dataset='openwebtext', valid_dataset='wikitext103', batch_size=args.batch_size, cache_dir=args.cache_dir, block_size=args.block_size, distributed=args.distributed, n_proc=64)
    # Get wikitext2
    args.dataset_name = 'wikitext2'
    wikitext2_dataloader = data.get_valid_dataloaders(dataset_name=args.dataset_name, batch_size=args.batch_size, cache_dir=args.cache_dir, block_size=args.block_size, distributed=args.distributed)
     # Get wikitext103
    args.dataset_name = 'wikitext103'
    wikitext103_dataloader = data.get_valid_dataloaders(dataset_name=args.dataset_name, batch_size=args.batch_size, cache_dir=args.cache_dir, block_size=args.block_size, distributed=args.distributed)   
     # Get ptb
    args.dataset_name = 'ptb'
    ptb_dataloader = data.get_valid_dataloaders(dataset_name=args.dataset_name, batch_size=args.batch_size, cache_dir=args.cache_dir, block_size=args.block_size, distributed=args.distributed)   
     # Get lambada
    args.dataset_name = 'lambada'
    lambada_dataloader = data.get_valid_dataloaders(dataset_name=args.dataset_name, batch_size=args.batch_size, cache_dir=args.cache_dir, block_size=args.block_size, distributed=args.distributed)   
     # Get lm1b
    args.dataset_name = 'lm1b'
    lm1b_dataloader = data.get_valid_dataloaders(dataset_name=args.dataset_name, batch_size=args.batch_size, cache_dir=args.cache_dir, block_size=args.block_size, distributed=args.distributed) 
    
    dataloader_dic = {
        'wikitext2': wikitext2_dataloader,
        'wikitext103': wikitext103_dataloader,
        'ptb': ptb_dataloader,
        'lambada': lambada_dataloader,
        'lm1b': lm1b_dataloader
    }
    for dataset_name in ['wikitext2', 'wikitext103', 'ptb', 'lambada', 'lm1b']:
        dataloader = dataloader_dic[dataset_name]
        eval_iter = iter(dataloader)
        total_loss = 0
        with torch.no_grad():
            batch_num = 0
            for batch in eval_iter:
                batch = batch['input_ids'].to(device)
                cur_loss = model(batch, labels = batch).loss
                total_loss += cur_loss
                batch_num += 1

        dist.all_reduce(total_loss)
        total_loss /= args.ngpus
        total_loss /= batch_num
        
        ppl = math.exp(total_loss)
        
        if master_process:
            print("================================")
            print(f'Dataset: {dataset_name}')
            print(f"Evaluation PPL: {ppl}")
            print("================================\n\n")
    dataloader = train_loader
    eval_iter = iter(dataloader)
    total_loss = 0
    with torch.no_grad():
        batch_num = 0
        for batch in eval_iter:
            batch = batch['input_ids'].to(device)
            cur_loss = model(batch, labels = batch).loss
            total_loss += cur_loss
            batch_num += 1
            if batch_num > 100:
                break

    dist.all_reduce(total_loss)
    total_loss /= args.ngpus
    total_loss /= batch_num
    
    ppl = math.exp(total_loss)
    
    if master_process:
        print("================================")
        print(f'Dataset: Openwentext')
        print(f'Loss: {total_loss}')
        print(f"Evaluation PPL: {ppl}")
        print("================================\n\n")    
    destroy_process_group()




if __name__ == "__main__":
    main()