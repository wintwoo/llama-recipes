#!/bin/bash

#SBATCH --job-name=llama2-convert-checkpoints-%u-%j
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=l4
#SBATCH --time=06:00:00

export MODEL_NAME="Llama-2-7b-chat-hf"
export FSDP_CHECKPOINT_PATH="/bucket/checkpoints/meta-llama/${MODEL_NAME}_new-meta-llama/${MODEL_NAME}"
export CONSOLIDATED_MODEL_PATH="${FSDP_CHECKPOINT_PATH}/save/checkpoints/"

cd $SLURM_SUBMIT_DIR

echo "converting FSDP checkpoint to HF checkpoints now"

time python -m llama_recipes.inference.checkpoint_converter_fsdp_hf --fsdp_checkpoint_path $FSDP_CHECKPOINT_PATH --consolidated_model_path $CONSOLIDATED_MODEL_PATH --HF_model_path_or_name $MODEL_NAME
