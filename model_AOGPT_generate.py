"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from catsample import sample_categorical, top_p_sampling_from_probs
import random
import time

@torch.compile
def add_scale_fused(x: Tensor, scale: Tensor, residual: Tensor) -> Tensor:
    return scale * x + residual 

@torch.compile
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:  
    return x * (1+scale) + shift

def _make_identity_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask.masked_fill_(torch.eye(tgt_len, device=device) != 0, 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for uni-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    
    mask_cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(mask_cond >= mask_cond.view(-1, 1), 0)
    
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

def _make_causal_with_parallel_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0,
    parallel_generated_tokens_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    if parallel_generated_tokens_length > 0:
        mask[-parallel_generated_tokens_length:, -parallel_generated_tokens_length:].masked_fill_(torch.eye(parallel_generated_tokens_length, device=device) == 0, torch.tensor(torch.finfo(dtype).min, device=device))
        
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)



class RMSNorm(nn.Module):
    def __init__(self, ndim):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return F.rms_norm(input, self.weight.shape, self.weight, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(0.)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.q_norm = RMSNorm(self.n_embd // self.n_head)
        self.k_norm = RMSNorm(self.n_embd // self.n_head)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence ; + 1 to handle the additional [None] token
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1)) 
                                        .view(1, 1, config.block_size + 1, config.block_size + 1))

    @torch.compile
    def forward(self, x, past_key_value, attn_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if past_key_value is None:
            past_key_value = torch.stack([k,v])
        else:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)
            past_key_value = torch.stack([k, v])
        

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0, is_causal=False)
        else:
            raise NotImplementedError
            # manual implementation of attention
            # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # att = F.softmax(att, dim=-1)
            # att = self.attn_dropout(att)
            # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, past_key_value

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    @torch.compile
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, 6 * config.n_embd, bias=True)
        )

    @torch.compile
    def forward(self, x, c, past_key_value, attn_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaLN(c)).chunk(6, dim=-1)
        attn_output, past_key_value = self.attn(modulate_fused(self.ln_1(x), shift_msa, scale_msa), past_key_value, attn_mask)
        x = add_scale_fused(attn_output, gate_msa, x)
        x = add_scale_fused(self.mlp(modulate_fused(self.ln_2(x), shift_mlp, scale_mlp)), gate_mlp, x)
        return x, past_key_value

class FinalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = RMSNorm(config.n_embd)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, 2 * config.n_embd, bias=True)
        )
    @torch.compile
    def forward(self, x, c):
        shift, scale = (self.adaLN_final(c)).chunk(2, dim=-1)
        x = modulate_fused(self.ln_f(x), shift, scale)
        return x

@dataclass
class AOGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class AOGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Add target position aware positional encoding
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size + 1, config.n_embd),  # + 1 to handle additional [None] token for unconditional generation
            wtpe = nn.Embedding(config.block_size, 128),     # [None] is not target   
            wnonee = nn.Embedding(1, config.n_embd),                   # embedding for [None] token
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            final_layer = FinalLayer(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.trunc_normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer), a=-3*0.02/math.sqrt(2 * config.n_layer), b=3*0.02/math.sqrt(2 * config.n_layer))

        self.kv_cache = True
        self.past_key_values = None

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        
    def get_kv_cache(self):
        return self.pask_key_values
    
    def get_kv_cache_len(self):
        return self.past_key_values.shape[-2]
    
    def clean_kv_cache(self):
        self.past_key_values = None
    
    def set_kv_cache(self, past_key_values):
        self.past_key_values = past_key_values
    
    def enable_kv_cache(self):
        self.kv_cache = True
        self.past_key_values = None
    
    def disable_kv_cache(self):
        self.kv_cache = False
        self.past_key_values = None

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-3*0.02, b=3*0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-3*0.02, b=3*0.02)
    
    # From rar https://github.com/bytedance/1d-tokenizer/blob/main/modeling/rar.py#L266
    def sample_random_orders(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        shuffled_orders = []

        for _ in range(batch_size):
            # random order
            shuffled_orders.append(torch.randperm(seq_length, device=x.device))
                
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)

    def sample_random_orders_CL(self, x, random_ratio):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        shuffled_orders = []

        for _ in range(batch_size):
            if random.random() < random_ratio:
                shuffled_orders.append(torch.randperm(seq_length, device=x.device))
            else:
                shuffled_orders.append(torch.arange(seq_length, device=x.device))
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)

    def set_ascending_orders(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        shuffled_orders = []

        for _ in range(batch_size):
            # ascending order
            shuffled_orders.append(torch.arange(seq_length, device=x.device))
                
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)
    
    # From rar https://github.com/bytedance/1d-tokenizer/blob/main/modeling/rar.py#L289
    def shuffle(self, x, orders):
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        shuffled_x = x[batch_indices, orders]
        return shuffled_x
    
    # From rar https://github.com/bytedance/1d-tokenizer/blob/main/modeling/rar.py#L295
    def unshuffle(self, shuffled_x, orders):
        # Unshuffle the tensor based on the original orders
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x
    
    def forward(self, idx, mode='Random', orders=None, random_ratio=None):
        if mode is None:
            assert orders is not None and idx.shape==orders.shape, 'mode is None, order should be given and with the same shape of idx'
        else:
            assert mode in ['AR', 'Random', 'Random_CL'], 'mode should be AR or Random or Random_CL'
        
        if mode is None:
            return self.forward_fn(idx, orders)
        elif mode == 'AR':
            # get ascending orders
            orders = self.set_ascending_orders(idx)
            return self.forward_fn(idx, orders)
        elif mode == 'Random':
            # get random orders
            orders = self.sample_random_orders(idx)   
            return self.forward_fn(idx, orders)
        elif mode == 'Random_CL':
            assert random_ratio is not None
            orders = self.sample_random_orders_CL(idx, random_ratio)
            return self.forward_fn(idx, orders)     
        

    def forward_fn_generate_first_step(self, target_positions):
        
        device = target_positions.device
        b, t = target_positions.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.zeros(t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wnonee(torch.tensor([[0]], device=device)) # [None] token embedding of shape (1, 1, n_embd)
        tok_emb = tok_emb.expand(b, t, -1) # expand to shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos).unsqueeze(0).expand(b, -1, -1) # positional embeddings of shape (b, t, n_embd)
        target_pos_emb = self.transformer.wtpe(target_positions) # target positional embeddings of shape (b, t, n_embd)
        
        # get attention mask
        attn_mask = _make_identity_mask((b,t), dtype=tok_emb.dtype, device=device)
        
        x = tok_emb + pos_emb
        
        
        # forward the GPT model itself
        x = self.transformer.drop(x)
        past_key_values_list = []
        for block in self.transformer.h:
            x, past_key_value = block(x, target_pos_emb, None, attn_mask)
            past_key_values_list.append(past_key_value)  
        past_key_values = torch.stack(past_key_values_list)
        self.past_key_values = torch.narrow(past_key_values, dim=4, start=0, length=1)
        x = self.transformer.final_layer(x, target_pos_emb)

        logits = self.lm_head(x)


        return logits
    
    def forward_fn_cond_cache_first_step(self, last_tokens, last_target_positions, target_positions):
        
        device = last_tokens.device
        b, t = last_tokens.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        tok_emb = self.transformer.wte(last_tokens) # token embeddings of shape (b, t, n_embd); this is the shuffled token embedding because the idx has been shuffled
        pos_emb = self.transformer.wpe(last_target_positions + 1)
        # TODO here
        none_tok_emb = self.transformer.wnonee(torch.tensor([[0]], device=last_tokens.device)) # [None] token embedding of shape (1, 1, n_embd)
        none_tok_emb = none_tok_emb.expand(b, -1, -1) # expand to shape (b, 1, n_embd)
        none_pos_emb = self.transformer.wpe(torch.tensor([[0]], device=last_tokens.device)).expand(b, -1, -1) # expand to shape (b, 1, n_embd)
        tok_emb = torch.cat([none_tok_emb, tok_emb], dim = 1)
        pos_emb = torch.cat([none_pos_emb, pos_emb], dim = 1)
        target_positions = torch.cat([last_target_positions, target_positions], dim = 1)
        target_pos_emb = self.transformer.wtpe(target_positions) # target positional embeddings of shape (b, t+1, n_embd)
        
        # get attention mask
        attn_mask = _make_causal_mask((b,t+1), dtype=tok_emb.dtype, device=device)
        
        x = tok_emb + pos_emb
        
        
        # forward the GPT model itself
        x = self.transformer.drop(x)
        past_key_values_list = []
        for block in self.transformer.h:
            x, past_key_value = block(x, target_pos_emb, None, attn_mask)
            past_key_values_list.append(past_key_value)  
        past_key_values = torch.stack(past_key_values_list)
        self.past_key_values = past_key_values
        # x = self.transformer.final_layer(x, target_pos_emb)

        # logits = self.lm_head(x)


        return None

    def forward_fn_generate(self, last_tokens, last_positions, target_positions):
        
        device = last_positions.device
        b, t_last = last_positions.size()
        _, t_target = target_positions.size()
        t_total = t_last + t_target - 1
        
        # print('t_last:', t_last)
        # print('t_target:', t_target)
        # print('last_tokens_len:', last_tokens.shape[-1])
        
        assert t_total <= self.config.block_size, f"Cannot forward sequence of length {t_total}, block size is only {self.config.block_size}"
        
        # token and positional embedding
        idx = torch.cat([last_tokens, torch.tile(last_tokens[:, -1:], (1,t_target-1))], dim=1)
        pos = torch.cat([last_positions, torch.tile(last_positions[:, -1:], (1,t_target-1))], dim=1)
        target_pos = torch.cat([last_positions[:,1:], target_positions], dim=1)

        tok_emb = self.transformer.wte(idx)
        # Here +1 to handle the additional first pe tokens
        pos_emb = self.transformer.wpe(pos + 1)
        target_pos_emb = self.transformer.wtpe(target_pos)
        
        # get attention mask
        kv_cache_len = self.get_kv_cache_len()
        attn_mask = _make_causal_with_parallel_mask((b,t_total), dtype=tok_emb.dtype, device=device, past_key_values_length=kv_cache_len, parallel_generated_tokens_length=t_target)
        
        # print('tok_emb.shape:', tok_emb.shape)
        # print('pos_emb.shape:', pos_emb.shape)
        # print('target_pos_emb.shape:', target_pos_emb.shape)
        x = tok_emb + pos_emb
        
        # forward the GPT model itself
        x = self.transformer.drop(x)
        past_key_values_list = []
        for i, block in enumerate(self.transformer.h):
            x, past_key_value = block(x, target_pos_emb, self.past_key_values[i], attn_mask)
            past_key_values_list.append(past_key_value)  
        past_key_values = torch.stack(past_key_values_list)
        self.past_key_values = torch.narrow(past_key_values, dim=4, start=0, length=kv_cache_len + t_last)
        # minioptimization here; only need to forward final t_target tokens
        x = x[:, -t_target:]
        target_pos_emb = target_pos_emb[:, -t_target:]
        x = self.transformer.final_layer(x, target_pos_emb)

        logits = self.lm_head(x)

        return logits
        
    
    @torch.no_grad()
    def generate(self, num_samples, sampling_steps, device, schedules='linear', topp = 0.9, temperature = 0.9, sampling_orders=None):
        assert schedules in ['linear', 'cosine'], f'{schedules} is not supported yet'
        schedule_fn = None
        if schedules == 'linear':
            schedule_fn = lambda t: t
        elif schedules == 'cosine':
            schedule_fn = lambda t: torch.cos((1-t) * math.pi * 0.5)
        timesteps = schedule_fn(torch.linspace(1, 0, sampling_steps+1, device=device)) # 1 0.5 0 to 1 0.7071 0
        
        if sampling_orders is None:
            sampling_orders = torch.stack([torch.randperm(self.config.block_size, device=device) for _ in range(num_samples)])
        # step 1
        # print('Step:', 0)
        num_current_generated_tokens = 0
        # print('num_current_generated_tokens:', num_current_generated_tokens)
        num_next_generated_tokens = max(int(1024*(1-timesteps[1])), num_current_generated_tokens+1)
        # print('num_next_generated_tokens:', num_next_generated_tokens)
        target_positions = sampling_orders[:, :num_next_generated_tokens]
        logits = self.forward_fn_generate_first_step(target_positions)
        # print('logits:', logits)
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        tokens = top_p_sampling_from_probs(probs, p=topp)
        # tokens = sample_categorical(probs, dtype=torch.bfloat16)
        # print('probs.shape:', probs.shape)
        # tokens = torch.multinomial(probs.squeeze(1), 1)
        # print('tokens:', tokens)
        total_tokens = tokens
        last_target_positions = target_positions
        last_tokens = tokens
        print(self.past_key_values.shape)
        
        for i in range(1, sampling_steps):
            # print('Step:', i)
            num_current_generated_tokens = num_next_generated_tokens
            num_next_generated_tokens = max(int(1024*(1-timesteps[i+1])), num_current_generated_tokens+1)
            target_positions = sampling_orders[:, num_current_generated_tokens:num_next_generated_tokens]
            logits = self.forward_fn_generate(last_tokens, last_target_positions, target_positions)
            # if i == 1:
                # print(logits)
            # print('logits.dtype:', logits.dtype)
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            # print('probs.dtype:', probs.dtype)
            tokens = top_p_sampling_from_probs(probs, p=topp)
            # tokens = sample_categorical(probs, dtype=torch.bfloat16)
            total_tokens = torch.cat([total_tokens, tokens], dim=1)
            last_target_positions = target_positions
            last_tokens = tokens

        return self.unshuffle(total_tokens, sampling_orders)
    
    @torch.no_grad()
    def cond_generate(self, num_samples, sampling_steps, device, schedules='linear', topp = 0.9, temperature = 0.9, input_ids=None, input_locs=None, sampling_orders=None):
        assert schedules in ['linear', 'cosine'], f'{schedules} is not supported yet'
        schedule_fn = None
        if schedules == 'linear':
            schedule_fn = lambda t: t
        elif schedules == 'cosine':
            schedule_fn = lambda t: torch.cos((1-t) * math.pi * 0.5)
        timesteps = schedule_fn(torch.linspace(1, 0, sampling_steps+1, device=device)) # 1 0.5 0 to 1 0.7071 0
        
        input_ids = input_ids.to(device)
        input_locs = input_locs.to(device)
        if sampling_orders is None:
            all_numbers = torch.arange(self.config.block_size).to(device)
            mask = ~torch.isin(all_numbers, input_locs)
            missing = all_numbers[mask]
            # sampling_orders = torch.stack([torch.cat([input_locs, missing[torch.randperm(missing.size(0))]]) for _ in range(num_samples)]).to(device)
            sampling_orders = torch.stack([torch.cat([input_locs, missing[torch.arange(missing.size(0))]]) for _ in range(num_samples)]).to(device)
        # print(sampling_orders[:, :100])

        # step 1
        # print('Step:', 0)
        len_input_ids = len(input_ids)
        input_ids = input_ids.unsqueeze(0).expand(num_samples, -1)
        input_locs = input_locs.unsqueeze(0).expand(num_samples, -1)
        num_initial_tokens = len_input_ids - 1
        num_current_generated_tokens = len_input_ids - 1
        # print('num_current_generated_tokens:', num_current_generated_tokens)
        last_tokens = input_ids[:, :-1]
        last_target_positions = input_locs[:, :-1]
        target_positions = input_locs[:, -1:]
        _ = self.forward_fn_cond_cache_first_step(last_tokens, last_target_positions, target_positions)
        print(self.past_key_values.shape)


        total_tokens = input_ids
        last_target_positions = input_locs[:, -1:]
        last_tokens = input_ids[:, -1:]
        num_next_generated_tokens = len_input_ids
        
        for i in range(1, sampling_steps+1):
            # print('Step:', i)
            num_current_generated_tokens = num_next_generated_tokens
            num_next_generated_tokens = max(len_input_ids + int((1024-len_input_ids)*(1-timesteps[i])), num_current_generated_tokens+1)
            target_positions = sampling_orders[:, num_current_generated_tokens:num_next_generated_tokens]
            logits = self.forward_fn_generate(last_tokens, last_target_positions, target_positions)
            # if i == 1:
                # print(logits)
            # print('logits.dtype:', logits.dtype)
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            # print('probs.dtype:', probs.dtype)
            tokens = top_p_sampling_from_probs(probs, p=topp)
            # tokens = sample_categorical(probs, dtype=torch.bfloat16)
            total_tokens = torch.cat([total_tokens, tokens], dim=1)
            last_target_positions = target_positions
            last_tokens = tokens
            print(self.past_key_values.shape)
        # print('total_tokens.shape:', total_tokens.shape)
        # print('sampling_orders.shape:', sampling_orders.shape)    
            
        return self.unshuffle(total_tokens, sampling_orders)
    

    def forward_fn_ppl(self, idx, orders):
        
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t+1, dtype=torch.long, device=device) # shape (t+1) to include the [None] token
        
        # shuffle input ids given orders
        idx = self.shuffle(idx, orders) # of shape (b, t)
        targets = idx # of shape (b, t)

        # prepare token embedding, position embedding, target position embedding and shuffle them
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd); this is the shuffled token embedding because the idx has been shuffled
        none_tok_emb = self.transformer.wnonee(torch.tensor([[0]], device=idx.device)) # [None] token embedding of shape (1, 1, n_embd)
        none_tok_emb = none_tok_emb.expand(idx.shape[0], -1, -1) # expand to shape (b, 1, n_embd)
        tok_emb = torch.cat([none_tok_emb, tok_emb], dim = 1) # concat to shape (b, t+1, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t+1, n_embd)
        pos_emb = pos_emb.unsqueeze(0).expand(idx.shape[0], -1, -1) # expand to shape (b, t+1, n_embd)
        pos_emb_prefix = pos_emb[:,:1] # position embedding prefix on the [None] token of shape (b, 1, n_embd)
        pos_emb_postfix = self.shuffle(pos_emb[:,1:], orders) # position embedding postfix of shape (b, t, n_embd); shuffled
        target_pos_emb = self.transformer.wtpe(pos[:t]) # target position embeddings of shape (t, n_embd)
        target_pos_emb = target_pos_emb.unsqueeze(0).expand(idx.shape[0], -1, -1) # expand to shape (b, t, n_embd)
        target_pos_emb_prefix = self.shuffle(target_pos_emb, orders) # shuffle target position embeddings of shape (b, t, n_embd)
        target_pos_emb_postfix = torch.zeros_like(target_pos_emb[:, :1]) # zeros of shape (b, 1, n_embd)
        target_pos_emb_final = torch.cat([target_pos_emb_prefix, target_pos_emb_postfix], dim = 1)
        
        x = tok_emb + torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)
        
        # forward the GPT model itself
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, target_pos_emb_final)
        x = self.transformer.final_layer(x, target_pos_emb_final)


        # if we are given some desired targets also calculate the loss
        logits = self.lm_head(x)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets
        loss = F.cross_entropy(shift_logits.permute(0,2,1), shift_targets, ignore_index=-1, reduction='none')


        return logits, loss

    def forward_fn(self, idx, orders):
        
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t+1, dtype=torch.long, device=device) # shape (t+1) to include the [None] token
        
        # shuffle input ids given orders
        idx = self.shuffle(idx, orders) # of shape (b, t)
        targets = idx # of shape (b, t)

        # prepare token embedding, position embedding, target position embedding and shuffle them
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd); this is the shuffled token embedding because the idx has been shuffled
        none_tok_emb = self.transformer.wnonee(torch.tensor([[0]], device=idx.device)) # [None] token embedding of shape (1, 1, n_embd)
        none_tok_emb = none_tok_emb.expand(idx.shape[0], -1, -1) # expand to shape (b, 1, n_embd)
        tok_emb = torch.cat([none_tok_emb, tok_emb], dim = 1) # concat to shape (b, t+1, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t+1, n_embd)
        pos_emb = pos_emb.unsqueeze(0).expand(idx.shape[0], -1, -1) # expand to shape (b, t+1, n_embd)
        pos_emb_prefix = pos_emb[:,:1] # position embedding prefix on the [None] token of shape (b, 1, n_embd)
        pos_emb_postfix = self.shuffle(pos_emb[:,1:], orders) # position embedding postfix of shape (b, t, n_embd); shuffled
        target_pos_emb = self.transformer.wtpe(pos[:t]) # target position embeddings of shape (t, n_embd)
        target_pos_emb = target_pos_emb.unsqueeze(0).expand(idx.shape[0], -1, -1) # expand to shape (b, t, n_embd)
        target_pos_emb_prefix = self.shuffle(target_pos_emb, orders) # shuffle target position embeddings of shape (b, t, n_embd)
        target_pos_emb_postfix = torch.zeros_like(target_pos_emb[:, :1]) # zeros of shape (b, 1, n_embd)
        target_pos_emb_final = torch.cat([target_pos_emb_prefix, target_pos_emb_postfix], dim = 1)
        
        x = tok_emb + torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)
        
        # forward the GPT model itself
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, target_pos_emb_final)
        x = self.transformer.final_layer(x, target_pos_emb_final)


        # if we are given some desired targets also calculate the loss
        logits = self.lm_head(x)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=-1)


        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size+1])   # + 1 to handle the additional [None] token
        self.transformer.wtpe.weight = nn.Parameter(self.transformer.wtpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size + 1,:block_size + 1] # + 1 to handle the additional [None] token

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size + 1  # + 1 to handle the additional [None] token
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # @torch.no_grad()
    # def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    #     """
    #     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    #     the sequence max_new_tokens times, feeding the predictions back into the model each time.
    #     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    #     """
    #     for _ in range(max_new_tokens):
    #         # if the sequence context is growing too long we must crop it at block_size
    #         idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    #         # forward the model to get the logits for the index in the sequence
    #         logits, _ = self(idx_cond)
    #         # pluck the logits at the final step and scale by desired temperature
    #         logits = logits[:, -1, :] / temperature
    #         # optionally crop the logits to only the top k options
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float('Inf')
    #         # apply softmax to convert logits to (normalized) probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         # append sampled index to the running sequence and continue
    #         idx = torch.cat((idx, idx_next), dim=1)

    #     return idx
