import os
import torch

# os.environ['CUDA_VISIBLE_DEVICES']='3'
# os.environ["RANK"]='0'
# os.environ['LOCAL_RANK']='0'
# os.environ['WORLD_SIZE']='1'
# os.environ['LOCAL_WORLD_SIZE']='1'
# os.environ["MASTER_ADDR"]='127.0.0.1'
# os.environ["MASTER_PORT"]="22111"

from uni_interleaved.models.utils.monkey_patch import (
    replace_llama_attn_with_flash_attn,
    replace_blip2_attn_with_qknorm_attn,
    replace_beam_search,
    replace_stable_diffusion_pipeline_call,
    replace_stable_diffusion_unet_forward,
    replace_logger_verbose
)

replace_beam_search()
replace_blip2_attn_with_qknorm_attn()
replace_stable_diffusion_unet_forward()
replace_stable_diffusion_pipeline_call()
replace_logger_verbose()

IS_TRAIN = True
if IS_TRAIN:
    replace_llama_attn_with_flash_attn()


from transformers.trainer_utils import get_last_checkpoint

from uni_interleaved.models import MMInterleaved
from uni_interleaved.custom_datasets.utils.build import build_train_dataset,build_eval_dataset
from uni_interleaved.engine.lmm_trainer import LMMTrainer
from uni_interleaved.utils import ArgumentParser, TrainingArguments, init_distributed_mode, load_model_weights


def main():
    parser = ArgumentParser(TrainingArguments)
    init_distributed_mode()
    args = parser.parse_args_with_config_file_into_dataclasses()
    train_args, config = args
    if train_args.load_from_args is not None:
        setattr(config, "load_from", train_args.load_from_args)
    print(train_args)
    print(config)

    print("Data Loading Start ####################################################")
    train_dataset = build_train_dataset(config.data.train)
    print(train_dataset)

    # eval_dataset = build_eval_dataset(config.data.val)
    # print(eval_dataset)
    eval_dataset=None

    print("Model Init Start ######################################################")
    model = MMInterleaved(**config.model)
    # print(model)

    print("Trainer Init Start #####################################################")
    trainer = LMMTrainer(
        model=model,
        tokenizer=train_dataset.tokenizer,
        config=config,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=train_dataset.collator,
        eval_dataset=eval_dataset,
        eval_collator=None,
    )

    if getattr(config, "load_from", None):
        load_model_weights(trainer.model, config.load_from)
    
    print("Training Start")
    trainer.train(
        resume_from_checkpoint=get_last_checkpoint(train_args.output_dir)
        if train_args.resume
        else None
    )
    trainer.save_state()
    trainer.save_model(output_dir=os.path.join(train_args.output_dir, "training_end"))
    print("All Finished")


if __name__ == "__main__":
    main()
