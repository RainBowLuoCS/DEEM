#! /bin/bash


export CUDA_DEVICE_MAX_CONNECTIONS=1

NNODES=$WORLD_SIZE  # Adjust
GPUS_PER_NODE=8 # Adjust

GPU_NUM=$(($GPUS_PER_NODE*$NNODES))
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "================================================"
echo "GPU_NUM: $GPU_NUM"
echo "================================================"


DISTRIBUTED_ARGS="\
              --nproc_per_node $GPUS_PER_NODE \
              --nnodes $NNODES \
              --node_rank $RANK \
              --master_addr $MASTER_ADDR \
              --master_port $MASTER_PORT \
"

echo $DISTRIBUTED_ARGS


mkdir -p OUTPUT/pretrain_convnext_base

torchrun $DISTRIBUTED_ARGS \
        train.py \
        --config_file=uni_interleaved/configs/train/pretrain.yaml \
        --output_dir=OUTPUT/pretrain_convnext_base |tee -a OUTPUT/pretrain_convnext_base/pretrain_convnext_base_train.log