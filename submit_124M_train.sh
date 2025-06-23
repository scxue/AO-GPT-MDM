#!/bin/bash
#SBATCH --job-name=AOGPT-S             # Job name
#SBATCH --nodes=2                      # Number of nodes
#SBATCH --cpus-per-task=224            # Number of CPU cores per task
#SBATCH --gres=gpu:8                   # Number of GPUs per node
#SBATCH --time=8-24:00:00              # Time limit (e.g., 24 hours)
#SBATCH --mem=256GB                    # Memory per node (adjust as needed)
#SBATCH --partition=your partition         # Partition name
#SBATCH --output=slurm_logs/AOGPT_S_%j.log        # Standard output and error log
#SBATCH --error=slurm_logs/AOGPT_S_%j.err


# Load necessary modules
conda activate nanogpt

# Define variables for torchrun
# MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 50000))                   # Specify a port number
NNODES=$SLURM_NNODES
GPUS_PER_NODE=8

# Print some debug information
echo "Master Node: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Number of Nodes: $NNODES"
echo "GPUs per Node: $GPUS_PER_NODE"
echo "Node Rank: $SLURM_PROCID$"


srun torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --rdzv_id $RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_AOGPT_CL.py config_larger_model_v3/train_AOGPT_AdaLN6_NoRep_cond_128.py


