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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cache_dir", type=str, default='/home/cache')
    parser.add_argument("--block_size", type=int, default=1024)
    
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    args = parser.parse_args()
    args.distributed = (ddp_world_size > 1)
    args.ngpus = ddp_world_size
    
    from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig
    model_args = dict(n_layer=24, n_head=16, n_embd=1024, block_size=args.block_size,
                  bias=True, vocab_size=None, dropout=0.0) # start with model_args from command line
    ckpt_dir = '/home/ckpt/350M/350M-AdaLN6_NoRep_cond_128_trunc_bs1M_wd_0.03_drop_0.0_CL_0.1_qknorm/ckpt_460000.pt'
    
    
    print(f"Resuming training from {ckpt_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(ckpt_dir)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    AOGPTconf = AOGPTConfig(**model_args)
    model = AOGPT(AOGPTconf)
    # state_dict = checkpoint['model']
    ema_params = checkpoint['ema_model']['shadow_params']
    for s_param, param in zip(ema_params, model.parameters()):
        param.data.copy_(s_param.data)
    model.to(device)
        

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
    ptb_dataloader = data.get_valid_dataloaders(dataset_name=args.dataset_name, batch_size=11, cache_dir=args.cache_dir, block_size=args.block_size, distributed=args.distributed)   
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
    for dataset_name in ['ptb']:
    # for dataset_name in ['wikitext2', 'wikitext103', 'ptb', 'lambada', 'lm1b']:
        dataloader = dataloader_dic[dataset_name]
        eval_iter = iter(dataloader)
        print('dataset:', dataset_name)
        print('batches:', len(eval_iter))
        total_loss = 0
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                batch_num = 0
                for j, batch in enumerate(eval_iter):
                    print(j)
                    _, cur_loss = model(batch['input_ids'].to(device), mode = 'AR')
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
    destroy_process_group()


if __name__ == "__main__":
    main()