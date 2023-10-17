#!/bin/bash

# We grab the first node address to act as the head node for multi node training
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=127.0.0.1
head_node_port=29500
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$head_node_port
export MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
export MODEL_NAME_NEW="${MODEL_NAME}_new"
#export TEMP=/scratch # this will use local SSDs configured for A2 VMs in the A100 slurm partition

# debugging flags (optional)
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export LOGLEVEL=INFO
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,P2P
#export NCCL_ASYNC_ERROR_HANDLING=1
#export TORCH_CPP_LOG_LEVEL=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "setting HF token to download base model"
#HUGGINGFACE_TOKEN="<YOUR TOKEN>"
HUGGINGFACE_TOKEN="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"
huggingface-cli login --token $HUGGINGFACE_TOKEN

#echo "starting training now.. on ${SLURM_NNODES} nodes with ${SLURM_GPUS} GPUs each"

#cd $SLURM_SUBMIT_DIR

time torchrun --nnodes 1 \
	--nproc_per_node 8 \
	--rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
	--rdzv-id foob \
	--rdzv-backend c10d \
	--log_dir logs \
	finetuning.py --model_name "$MODEL_NAME" --use_peft true --peft_method lora --quantization false --batch_size_training 1 --num_epochs 1 --dist_checkpoint_root_folder /bucket/model_checkpoints --dist_checkpoint_folder "$MODEL_NAME_NEW"
