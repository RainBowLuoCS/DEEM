import os.path as osp
import json
import random
import torch
import numpy as np
import glob

from ..utils.loader import BaseDataset
from ..utils.wds_utils import init_tokenizer

class ImageNetEvalCollator:
    def __init__(
        self,
        tokenizer_path,
        num_img_token=32,
        generation_kwargs=None,
        instr_prompts=None,
        train_dataset=None,
    ):
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.num_img_token = num_img_token

        default_generation_kwargs = dict(
            max_length=10,
            min_length=0,
            length_penalty=0.,
            num_beams=3,
            top_p=1.0,
        )
        self.generation_kwargs = generation_kwargs or default_generation_kwargs

        default_instr_prompts = [
            "## ASSISTANT: The answer is",
            "## USER: Based on the image, please answer the question. {image}{question} \n",
            "You are a helpful assistant.\n\n",
        ]

        self.instr_prompts = instr_prompts or default_instr_prompts
        self.train_dataset = train_dataset

        self.image_subseq = "<|sniffer|>" * self.num_img_token
        self.image_subseq = "<|startofimage|>" + self.image_subseq

    def __call__(self, data_list):
        return self._call_for_generate_texts(data_list)

    def _call_for_generate_texts(self, data_list):

        meta = []
        images_tensors_all = []
        images_tensors_dec_all = []
        num_image_per_seq = []
        text_inputs_with_prompt_image_all = []

        assis_prompt, user_prompt, sys_prompt = self.instr_prompts

        assert "{image}" in user_prompt and "{question}" in user_prompt

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor, question, answer, index, path = data

            assert isinstance(images_tensor, tuple)
            images_tensor, images_tensor_dec = images_tensor

            images_tensor = torch.from_numpy(images_tensor)
            images_tensor_dec = torch.from_numpy(images_tensor_dec)

            meta.append((index, question, answer, path))

            _images_tensor_all = [images_tensor]
            _image_tensors_dec_all = [images_tensor_dec]
            _num_image_per_seq = 1

            text_input = user_prompt.format(
                image=self.image_subseq, question=question
            )

            text_input = f"{sys_prompt} {text_input} {assis_prompt}".strip()

            images_tensors_all.extend(_images_tensor_all)
            images_tensors_dec_all.extend(_image_tensors_dec_all)

            num_image_per_seq.append(_num_image_per_seq)
            text_inputs_with_prompt_image_all.append(text_input)


        self.tokenizer.padding_side = "left"
        text_tensor = self.tokenizer(
            text_inputs_with_prompt_image_all,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = torch.stack(images_tensors_all, dim=0)
        images_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)
        
        assert images_tensors_dec.shape[0] == images_tensors.shape[0]

        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )

        data = dict(
            image_tensors=images_tensors,
            image_tensors_mask=None,
            image_tensors_dec=images_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            loss_img_weight=0.0,
            ignore_prompt_token_offset=ignore_prompt_token_offsets,
            meta=meta,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

class ImageNetEvalDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_file,
        transform,
        tokenizer_path,
        total_length=None,
        num_img_token=32,
        collate_mode='generate_imagenet',
        phase="a",
    ) -> None:
        '''
            Imagenet dataset only for acc
        '''
        super().__init__()

        assert phase in ('a','r','s','v2','d')

        self.phase = phase
        self.transform = transform
        self.data_root = data_root
        self.annt_file = annt_file
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.num_img_token = num_img_token
        self.collate_mode = collate_mode

        
        if phase=='a':
            self.annts=json.load(open(self.annt_file,"r"))['imagenet_a']
        elif phase=='r':
            self.annts=json.load(open(self.annt_file,"r"))['imagenet_r']
        elif phase=='s':
            self.annts=json.load(open(self.annt_file,"r"))['imagenet_s']
        elif phase=='d':
            self.annts=json.load(open(self.annt_file,"r"))['imagenet_d']
        else:
            self.annts=json.load(open(self.annt_file,"r"))['imagenet_v2']
        if phase not in ('s','d'):
            annts=[]
            for i in self.annts:
                annts.append({"file_name":i['file_name'],"caption":i["gt_caption"],'answer': 'yes'})
                annts.append({"file_name":i['file_name'],"caption":i["error_caption"],'answer': 'no'})
            self.annts=annts      
        random.seed(8823)
        random.shuffle(self.annts)
        # quick eval
        # self.annts=self.annts[:2000]
        

    def __repr__(self) -> str:
        return (
            f"ImageNet Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_file} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):

        info=self.annts[index]

        image_path = osp.join(self.data_root,info['file_name'])

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)
        if self.phase in ('d','s'):
            choise=[info['error_caption'],info['gt_caption']]
            random.shuffle(choise)
            question = f"What is the main object in this image? Choose from the following list:[{choise[0]}, {choise[1]}]"
            answer=f"{info['gt_caption']}"
        else:
            question = f"Is {info['caption']} the main object in this image? Please anwser yes or no"
            # question = f"Is {info['caption']} the main object in this image? Please anwser yes or no"
            answer=f"{info['answer']}"

        return image, question, answer, index, image_path
        
class POPEEvalDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_file,
        transform,
        tokenizer_path,
        total_length=None,
        num_img_token=32,
        collate_mode='generate_imagenet',
        phase="popular",
    ) -> None:
        '''
            Imagenet dataset only for NDCG evaluation
        '''
        super().__init__()

        assert phase in ('random','adversarial','popular')

        self.phase = phase
        self.transform = transform
        self.data_root = data_root
        self.annt_file = annt_file
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.num_img_token = num_img_token
        self.collate_mode = collate_mode

        self.annts=[json.loads(q) for q in open(osp.join(self.annt_file,'coco_pope_{}.json'.format(phase)), 'r')]
        
        # random.seed(8823)
        # random.shuffle(self.annts)
        # self.annts=self.annts[:60000]
        

    def __repr__(self) -> str:
        return (
            f"POPE Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_file} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):

        info=self.annts[index]

        image_path = osp.join(self.data_root,info['image'])

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)
        
        question = f"{info['text']} Please anwser yes or no"
        answer=info['label']

        return image, question, answer, index, image_path
    
class VisEvalDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_file,
        transform,
        tokenizer_path,
        total_length=None,
        num_img_token=32,
        collate_mode='generate_imagenet',
        phase="popular",
    ) -> None:
        '''
            Imagenet dataset only for visualization evaluation
        '''
        super().__init__()

        self.phase = phase
        self.transform = transform
        self.data_root = data_root
        self.annt_file = annt_file
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.num_img_token = num_img_token
        self.collate_mode = collate_mode

        self.annts=json.load(open("imagenet-a_vis.json","r"))
        # self.annts=[{"image_path": "./vis/vis/cat.png"},
        #             {"image_path": "./vis/vis/flower1.png"},
        #             {"image_path": "./vis/vis/tiger.png"},
        #             {"image_path": "./vis/vis/cat.png"},
        #             {"image_path": "./vis/vis/flower2.png"},
        #             {"image_path": "./vis/vis/dog.png"},
        #             ]
        
        

    def __repr__(self) -> str:
        return (
            f"Vis Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_file} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):

        info=self.annts[index][0]
        # info=self.annts[index]

        try:
            image = self.loader(info['image_path']).convert("RGB")

            image = self.transform(image)
        except:
            print(info['image_path'])
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)
        
        question = f"Please anwser yes or no"
        answer=" "

        return image, question, answer, index, info['image_path']