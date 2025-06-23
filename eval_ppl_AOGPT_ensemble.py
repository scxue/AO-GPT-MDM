import torch
import data
import argparse
import os
import torch.distributed as dist
import math
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from torch.distributed import init_process_group, destroy_process_group
import time


def main():
    parser = argparse.ArgumentParser(description="Evaluation Perplexity")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/cache")
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
    
    from model_AOGPT_generate import AOGPT, AOGPTConfig
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
    # for dataset_name in ['wikitext2', 'wikitext103', 'ptb', 'lambada', 'lm1b']:
    ensemble_times = 64
    for dataset_name in ['wikitext2', 'wikitext103', 'ptb', 'lambada', 'lm1b']:
        dataloader = dataloader_dic[dataset_name]
        eval_iter = iter(dataloader)
        print('dataset:', dataset_name)
        print('batches:', len(eval_iter))
        with torch.no_grad():
            ensemble_loss_list = []
            non_ensemble_loss_list = []
            for batch in eval_iter:

                batch = batch['input_ids'].to(device)
                orders = torch.stack([torch.randperm(args.block_size, device=device) for _ in range(batch.shape[0])])
                _, non_ensemble_loss = model.forward_fn_ppl(batch, orders=orders)
                non_ensemble_loss.to(torch.float64).to(device)
                ensemble_loss = torch.zeros([batch.shape[0], args.block_size, ensemble_times], dtype=torch.float64).to(device)
                for i in range(1024):
                    ensemble_loss[:, :, 0] = non_ensemble_loss
                    for j in range(1, ensemble_times):
                        reshuffle = torch.stack([torch.randperm(i, device=device) for _ in range(batch.shape[0])])
                        orders_ensemble = torch.cat((torch.gather(orders[:, :i], dim=1, index=reshuffle), orders[:, i:]), dim=1)
                        _, cur_loss_ensemble = model.forward_fn_ppl(batch, orders=orders_ensemble)
                        ensemble_loss[:, i, j] = cur_loss_ensemble[:, i]

                        
                ensemble_loss_list.append(ensemble_loss)
                non_ensemble_loss_list.append(non_ensemble_loss)
        
        final_ensemble_loss = torch.cat(ensemble_loss_list, dim = 0)
        final_non_ensemble_loss = torch.cat(non_ensemble_loss_list, dim = 0)
        if master_process:
            ensemble_gather_list = [torch.zeros_like(final_ensemble_loss) for _ in range(ddp_world_size)]
            nonensemble_gather_list = [torch.zeros_like(final_non_ensemble_loss) for _ in range(ddp_world_size)]
            dist.gather(final_ensemble_loss, ensemble_gather_list, 0)
            dist.gather(final_non_ensemble_loss, nonensemble_gather_list, 0)
        else:
            dist.gather(final_ensemble_loss, 0)
            dist.gather(final_non_ensemble_loss, 0)
        if master_process:
            total_ensemble_loss = torch.cat(ensemble_gather_list, dim = 0)
            total_non_ensemble_loss = torch.cat(nonensemble_gather_list, dim = 0)
            print('total_ensemble_loss.shape:', total_ensemble_loss)
            print('total_non_ensemble_loss.shape:', total_non_ensemble_loss)
            total_non_ensemble_ppl = torch.exp(torch.mean(total_non_ensemble_loss)).item()
            
            print("================================")
            print(f'Dataset: {dataset_name}')
            print(f"Nonensemble PPL: {total_non_ensemble_ppl}")
            print("================================")
            current_ensemble_time = 2
            while current_ensemble_time <= ensemble_times:
                total_ensemble_ppl = torch.exp(total_ensemble_loss[:,:,:current_ensemble_time].exp().reciprocal().mean(dim=-1).reciprocal().log().mean()).item()
                print(f"Ensemble PPL: {total_ensemble_ppl}")
                print(f"Ensemble times: {current_ensemble_time}")
                print("================================")
                current_ensemble_time *= 2
    dataloader = train_loader
    eval_iter = iter(dataloader)
    with torch.no_grad():
        ensemble_loss_list = []
        non_ensemble_loss_list = []
        for num_of_batches, batch in enumerate(eval_iter):
            if num_of_batches >= 128//ddp_world_size:
                break
            batch = batch['input_ids'].to(device)
            orders = torch.stack([torch.randperm(args.block_size, device=device) for _ in range(batch.shape[0])])
            _, non_ensemble_loss = model.forward_fn_ppl(batch, orders=orders)
            non_ensemble_loss.to(torch.float64).to(device)
            ensemble_loss = torch.zeros([batch.shape[0], args.block_size, ensemble_times], dtype=torch.float64).to(device)
            for i in range(1024):
                ensemble_loss[:, :, 0] = non_ensemble_loss
                for j in range(1, ensemble_times):
                    reshuffle = torch.stack([torch.randperm(i, device=device) for _ in range(batch.shape[0])])
                    orders_ensemble = torch.cat((torch.gather(orders[:, :i], dim=1, index=reshuffle), orders[:, i:]), dim=1)
                    _, cur_loss_ensemble = model.forward_fn_ppl(batch, orders=orders_ensemble)
                    ensemble_loss[:, i, j] = cur_loss_ensemble[:, i]

                    
            ensemble_loss_list.append(ensemble_loss)
            non_ensemble_loss_list.append(non_ensemble_loss)
    
    final_ensemble_loss = torch.cat(ensemble_loss_list, dim = 0)
    final_non_ensemble_loss = torch.cat(non_ensemble_loss_list, dim = 0)
    if master_process:
        ensemble_gather_list = [torch.zeros_like(final_ensemble_loss) for _ in range(ddp_world_size)]
        nonensemble_gather_list = [torch.zeros_like(final_non_ensemble_loss) for _ in range(ddp_world_size)]
        dist.gather(final_ensemble_loss, ensemble_gather_list, 0)
        dist.gather(final_non_ensemble_loss, nonensemble_gather_list, 0)
    else:
        dist.gather(final_ensemble_loss, 0)
        dist.gather(final_non_ensemble_loss, 0)
    if master_process:
        total_ensemble_loss = torch.cat(ensemble_gather_list, dim = 0)
        total_non_ensemble_loss = torch.cat(nonensemble_gather_list, dim = 0)
        print('total_ensemble_loss.shape:', total_ensemble_loss)
        print('total_non_ensemble_loss.shape:', total_non_ensemble_loss)
        total_non_ensemble_ppl = torch.exp(torch.mean(total_non_ensemble_loss)).item()
        
        print("================================")
        print(f'Dataset: Openwebtext')
        print(f'Nonensemble loss: {np.log(total_non_ensemble_ppl)}')
        print(f"Nonensemble PPL: {total_non_ensemble_ppl}")
        print("================================")
        current_ensemble_time = 2
        while current_ensemble_time <= ensemble_times:
            total_ensemble_ppl = torch.exp(total_ensemble_loss[:,:,:current_ensemble_time].exp().reciprocal().mean(dim=-1).reciprocal().log().mean()).item()
            print(f"Ensemble loss: {np.log(total_ensemble_ppl)}")
            print(f"Ensemble PPL: {total_ensemble_ppl}")
            print(f"Ensemble times: {current_ensemble_time}")
            print("================================")
            current_ensemble_time *= 2
    destroy_process_group()




if __name__ == "__main__":
    main()