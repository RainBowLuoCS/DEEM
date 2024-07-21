### pretrain

mkdir -p OUTPUT/pretrain_convnext_base

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=25641 train.py \
    --config_file=uni_interleaved/configs/train/pretrain.yaml \
    --output_dir=OUTPUT/pretrain_convnext_base |tee -a OUTPUT/pretrain_convnext_base/pretrain_convnext_base_train.log

# ### stage2
# mkdir -p OUTPUT/sft_referring_convnext_base

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=25641 train.py \
#     --config_file=uni_interleaved/configs/train/sft_referring.yaml \
#     --load_from_args=OUTPUT/sft_grounding_convnext_base/training_end/pytorch_model.bin \
#     --output_dir=OUTPUT/sft_referring_convnext_base |tee -a OUTPUT/sft_referring_convnext_base/sft_referring_convnext_base_train.log

# ### stage3
# mkdir -p OUTPUT/sft_vqa_convnext_base

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=25641 train.py \
#     --config_file=uni_interleaved/configs/train/sft_vqa.yaml \
#     --load_from_args=OUTPUT/sft_referring_convnext_base/training_end/pytorch_model.bin \
#     --output_dir=OUTPUT/sft_vqa_convnext_base |tee -a OUTPUT/sft_vqa_convnext_base/sft_vqa_convnext_base_train.log