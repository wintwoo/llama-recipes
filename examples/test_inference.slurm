#!/bin/bash

#SBATCH --job-name=llama2-inference-%u-%j
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=l4
#SBATCH --time=06:00:00


export MODEL_NAME="Llama-2-7b-chat-hf"
export FSDP_CHECKPOINT_PATH="/bucket/checkpoints/meta-llama/${MODEL_NAME}_new-meta-llama/${MODEL_NAME}"
export CONSOLIDATED_MODEL_PATH="${FSDP_CHECKPOINT_PATH}/save/checkpoints/"
export PROMPT_FILE="chat_completion/chats.json"

echo "Running inference using checkpoint stored in ${CONSOLIDATED_MODEL_PATH}"

cd $SLURM_SUBMIT_DIR
# --HF_model_path_or_name specifies the HF Llama model name or path where it has config.json and tokenizer.json
time python inference.py --model_name $CONSOLIDATED_MODEL_PATH --prompt_file $PROMPT_FILE --max_padding_length 1024
