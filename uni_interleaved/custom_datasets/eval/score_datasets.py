import os.path as osp
import json
import random
import torch
import numpy as np
import glob

from ..utils.loader import BaseDataset
from ..utils.wds_utils import init_tokenizer

class ScoreEvalCollator:
    def __init__(self):
        pass

    def __call__(self, data_list):
        image_ids = []
        images_tensors_all = []
        images_tensors_dec_all=[]
        context_ids = []
        context_attn_masks = []
        options_ids = []
        options_attn_masks = []
        # gt_relevances = []

        for data in data_list:
            image_ids.append(data["image_id"])
            images_tensor = data["image_tensor"]

            assert isinstance(images_tensor, tuple), images_tensor
            images_tensor, images_tensor_dec = images_tensor
            images_tensor = torch.from_numpy(images_tensor)
            images_tensors_all.append(images_tensor)
            images_tensor_dec = torch.from_numpy(images_tensor_dec)
            images_tensors_dec_all.append(images_tensor_dec)
            context_ids.append(data["text_ids"])
            context_attn_masks.append(data["attn_mask"])
            options_ids.append(data["options_ids"])
            options_attn_masks.append(data["options_attn_mask"])
            # gt_relevances.append(data['gt_relevance'])

        image_ids = torch.tensor(image_ids)
        images_tensors= torch.stack(images_tensors_all)
        images_tensors_dec = torch.stack(images_tensors_dec_all)
        num_image_per_seq = torch.ones(
            (images_tensors.shape[0],), dtype=torch.long, device=images_tensors.device
        )

        return dict(
            text_ids=context_ids,
            image_tensors=images_tensors,
            image_tensors_dec=images_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            attention_mask=context_attn_masks,
            # gt_relevances=gt_relevances,
            options_ids=options_ids,
            options_attn_masks=options_attn_masks,
            image_ids=image_ids,
        )

class VisDialDenseEvalDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        tokenizer_path,
        total_length=None,
        num_img_token=32,
        collate_mode='generate_scores',
        phase="val",
    ) -> None:
        '''
            VisDial dataset only for NDCG evaluation
        '''
        super().__init__()

        assert phase == 'val'

        self.phase = phase
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.num_img_token = num_img_token
        self.collate_mode = collate_mode

        dialog_json_path = osp.join(self.annt_root, 'visdial_1.0_val.json')
        with open(dialog_json_path, 'r') as rf:
            data = json.load(rf)["data"]
        
        self.dialogs = data["dialogs"]
        self.questions = data["questions"]
        self.answers = data["answers"]

        dense_annt_path = osp.join(self.annt_root, 'visdial_1.0_val_dense_annotations.json')
        with open(dense_annt_path, 'r') as rf:
            data_dense = json.load(rf)
        self.dense_annt = {d["image_id"]:d for d in data_dense}

        if total_length is not None:
            self.dialogs = self.dialogs[:total_length]
        print(f"length of the dataset is {len(self.dialogs)}")

    def __repr__(self) -> str:
        return (
            f"VisDial Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):

        item = self.dialogs[index]

        image_id = item["image_id"]
        image_path = osp.join(self.data_root, "VisualDialog_val2018", f"VisualDialog_val2018_{image_id:012d}.jpg")

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)
        
        image_prompt = "<|startofimage|>" + "<|sniffer|>" * self.num_img_token
        text = f"You are a helpful assistant.\n\n  ## USER: {image_prompt} caption: {item['caption']}. "
        dense_annt = self.dense_annt[image_id]
        round_idx = dense_annt["round_id"] - 1
        dialog = item["dialog"]
        for rnd in range(round_idx-1):
            question = self.questions[dialog[rnd]["question"]]
            answer = self.answers[dialog[rnd]["answer"]]
            text += f"question: {question}? answer: {answer}. "
        
        question = self.questions[dialog[round_idx]["question"]]
        text += f"question: {question}? \n ## ASSISTANT: the answer is"

        options = dialog[round_idx]["answer_options"]
        options = [self.answers[i] for i in options]
        # gt_relevance = dense_annt["gt_relevance"]

        # assert len(gt_relevance) == len(options)

        text_tensor = self.tokenizer(
            [text],
            truncation=False,
            padding=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids[0]
        attn_mask = text_tensor.attention_mask[0]

        options_tensor = self.tokenizer(
            options,
            truncation=False,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        options_ids = options_tensor.input_ids
        options_attn_mask = options_tensor.attention_mask

        return dict(
            image_id=image_id,
            image_tensor=image,
            # context=text,
            # options=options,
            text_ids=text_ids,
            attn_mask=attn_mask,
            options_ids=options_ids[:,1:], # no <bos>
            options_attn_mask=options_attn_mask[:,1:],
            # gt_relevance=gt_relevance,
        )
