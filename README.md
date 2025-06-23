
# AO-GPT-MDMD

This is the repository for training/infering an Masked Diffusion Model in a GPT-style based on [nanoGPT](https://github.com/karpathy/nanoGPT/).



## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## data preprocessing

```sh
bash submit_data_preprocess.sh
```
This bash script is to download and preprocess the necessary datasets (OpenWebText, Wikitext, 1BW, LAMBADA, etc.) before training.

## Train an AO-GPT

Train a GPT-2 Small scale model.
```sh
bash submit_124M_train.sh
```

Train a GPT-2 Medium scale model.
```sh
bash submit_350M_train.sh
```


## Pretrained Checkpoint


### Pretrained Checkpoints

My pretrained checkpoints for AO-GPT (Small, Medium) and Sigma-GPT (Small, Medium) are hosted on Hugging Face at [Cauthyyy/AO-GPT-MDM](https://huggingface.co/Cauthyyy/AO-GPT-MDM).

| Model              | Link                                                                                                                                                 |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AO-GPT-Small** | [Link](https://huggingface.co/Cauthyyy/AO-GPT-MDM/blob/main/124M/124M-AdaLN6_NoRep_cond_128_trunc_bs1M_wd_0.03_drop_0.0_CL_0.1_qknorm/ckpt_250000.pt) |
| **AO-GPT-Medium** | [Link](https://huggingface.co/Cauthyyy/AO-GPT-MDM/blob/main/350M/350M-AdaLN6_NoRep_cond_128_trunc_bs1M_wd_0.03_drop_0.0_CL_0.1_qknorm/ckpt_460000.pt) |
| **Sigma-GPT-Small**| [Link](https://huggingface.co/Cauthyyy/AO-GPT-MDM/blob/main/sigmaGPT/124M/ckpt_1000000.pt)                                                             |
| **Sigma-GPT-Medium**| [Link](https://huggingface.co/Cauthyyy/AO-GPT-MDM/blob/main/sigmaGPT/350M/ckpt_740000.pt)                                                              |

***



## sampling / inference


```sh
bash sample_AOGPT.sh
```

Try different sampling steps, Top-p, and temperature settings!







## acknowledgements

This repo is heavily built on [nanoGPT](https://github.com/karpathy/nanoGPT/).