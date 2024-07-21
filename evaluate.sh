### caption

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=25641 evaluate.py \
    --config_file=uni_interleaved/configs/eval/caption_eval.yaml \
    --load_from_args=OUTPUT/pretrain_convnext_base/training_end/pytorch_model.bin \
    --output_dir=OUTPUT/sft_vqa_convnext_base |tee -a OUTPUT/sft_vqa_convnext_base/sft_vqa_convnext_base_val.log

# ### referring

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
#   --nproc_per_node=4 --master_port=25641 evaluate.py \
#     --config_file=uni_interleaved/configs/eval/referring_eval.yaml \
#     --load_from_args=OUTPUT/pretrain_convnext_base/training_end/pytorch_model.bin \
#     --output_dir=OUTPUT/sft_referring_convnext_base |tee -a OUTPUT/sft_referring_convnext_base/sft_referring_convnext_base_eval.log

# ### robust

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
#   --nproc_per_node=4 --master_port=25641 evaluate.py \
#     --config_file=uni_interleaved/configs/eval/robust_eval.yaml \
#     --load_from_args=OUTPUT/pretrain_convnext_base/training_end/pytorch_model.bin \
#     --output_dir=OUTPUT/sft_vqa_convnext_base |tee -a OUTPUT/sft_vqa_convnext_base/sft_score_convnext_base_val.log

# ### vqa

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
#     --nproc_per_node=4 --master_port=25641 evaluate.py \
#     --config_file=uni_interleaved/configs/eval/vqa_eval.yaml \
#     --load_from_args=OUTPUT/pretrain_convnext_base/training_end/pytorch_model.bin \
#     --output_dir=OUTPUT/sft_vqa_convnext_base |tee -a OUTPUT/sft_vqa_convnext_base/sft_vqa_convnext_base_val.log