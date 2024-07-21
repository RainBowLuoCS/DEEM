import re
import os
import json
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from timm.data.transforms import RandomResizedCropAndInterpolation
import cv2

from ..utils.wds_utils import init_tokenizer
from ..utils.loader import BaseDataset

def load_json(url, client):
    with open(url, 'r') as file:
        data = json.load(file)
    return data

class GroundingEvalCollator:
    def __init__(
        self,
        tokenizer_path,
        task="grounding",
        num_img_token=32,
        generation_kwargs=None,
        instr_prompts=None,
        train_dataset = None,
        ignore_soi_token_loss=False,
        ignore_bos2soi_token_loss=False,
        max_length=2048,
    ):
        self.tasks=("grounding", "referring")

        assert task in self.tasks
        
        self.tokenizer = init_tokenizer(tokenizer_path, add_grounding_special_tokens=True)
        self.task = task
        self.num_img_token = num_img_token
        self.max_length=max_length
        self.train_dataset = train_dataset

        self.ignore_soi_token_loss = ignore_soi_token_loss
        self.ignore_bos2soi_token_loss = ignore_bos2soi_token_loss

        self.generation_kwargs = generation_kwargs

        default_instr_prompts={
            "grounding":[
                "## ASSISTANT: the bounding box coordinate is <box>",
                "## USER: {image} Provide the bounding box coordinate of the region this sentence describes: {caption}. \n",
                "You are a helpful assistant.\n\n",
            ],
             "referring":
            [
                "## ASSISTANT:",
                "## USER: {image} Provide a short description for this <ref>region1</ref> {mask}. \n",
                "You are a helpful assistant.\n\n",
            ]
        }

        self.instr_prompts = instr_prompts or default_instr_prompts

        self.image_subseq = "<|sniffer|>" * self.num_img_token
        self.image_subseq = "<|startofimage|>" + self.image_subseq

    def box2str(self, box):
        x1, y1, x2, y2 = box
        assert x1 <= x2 and y1 <= y2

        return f"({x1:03d},{y1:03d})({x2:03d},{y2:03d})"

    def __call__(self, data_list):
        
        return self._call_for_generate_texts(data_list)

    def _call_for_generate_texts(self, data_list):
        meta = []
        images_tensors_all = []
        images_tensors_dec_all=[]
        images_tensors_mask_all=[]
        num_image_per_seq = []
        text_inputs_with_prompt_image_all = []

        assis_prompt, user_prompt, sys_prompt=self.instr_prompts[self.task]

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor = data['images_tensor']
            assert isinstance(images_tensor, tuple), images_tensor

            question = data.get('query', 'Provide a short description for above region')  # None if self.task is not "region_vqa"
            answer = data['label']
            index = data['id']

            
            assert isinstance(images_tensor, tuple), images_tensor

            meta.append((index, question, answer, data['image'].height, data['image'].width, data['bbox']))

            images_tensor, images_tensor_dec = images_tensor
            images_tensor = torch.from_numpy(images_tensor)
            _images_tensor_all = [images_tensor]
            images_tensor_dec = torch.from_numpy(images_tensor_dec)
            _images_tensor_dec_all = [images_tensor_dec]
            _images_tensor_mask_all = [torch.ones(images_tensor.shape[-2:])[None,]]
            _num_image_per_seq=1

            # NOTE add object once again
            if self.task=='referring':
                _images_tensor_mask_all.append(torch.from_numpy(data['image_tensor_mask']))
                _images_tensor_all.append(images_tensor)
                _images_tensor_dec_all.append(images_tensor_dec)
                _num_image_per_seq=_num_image_per_seq+1

            if self.task == 'grounding':
                box = self.box2str(data['bbox'])
                text_input = user_prompt.format(
                    image=self.image_subseq, caption=answer,
                )
            elif self.task == "referring":
                text_input = user_prompt.format(
                    image=self.image_subseq, mask=self.image_subseq,
                )
            else:
                raise NotImplementedError

            text_input = f"{sys_prompt} {text_input} {assis_prompt}".strip()

            images_tensors_all.extend(_images_tensor_all)
            images_tensors_dec_all.extend(_images_tensor_dec_all)
            images_tensors_mask_all.extend(_images_tensor_mask_all)
            num_image_per_seq.append(_num_image_per_seq)

            text_inputs_with_prompt_image_all.append(text_input)

        self.tokenizer.padding_side ="left"
        text_tensor = self.tokenizer(
            text_inputs_with_prompt_image_all,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            max_length=self.max_length,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = torch.stack(images_tensors_all, dim=0)
        images_tensors_mask = torch.stack(images_tensors_mask_all, dim=0)
        images_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)

        assert images_tensors_dec.shape[0] == images_tensors.shape[0]
        assert images_tensors_mask.shape[0] == images_tensors.shape[0]

        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )

        assert num_image_per_seq.sum() == images_tensors.shape[0], (num_image_per_seq.sum(),images_tensors.shape[0])

        data = dict(
            image_tensors=images_tensors,
            image_tensors_mask=images_tensors_mask,
            image_tensors_dec=images_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            gt_text_ids=None,
            loss_img_weight=0.0,
            ignore_prompt_token_offset=ignore_prompt_token_offsets,
            meta=meta,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

class GroundingBaseEvalDataset(BaseDataset):
    def __init__(
        self,
        transform= None,
        box_scale: int = 999,
        collate_mode: str = 'generate_grounding',
        return_image: bool = True,
        random_flip: bool = False,
        random_resize_crop_prob: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ann = []
        self.box_scale = box_scale
        self.transform = transform
        self.resolution=self.transform.transform1.resolution
        self.collate_mode = collate_mode
        self.return_image = return_image

        self.random_flip = random_flip
        self.random_resize_crop_prob = random_resize_crop_prob
        self.grounded_caption_err = 0

        if self.random_resize_crop_prob > 0:
            self.random_resize_crop = RandomResizedCrop(self.resolution, interpolation='bicubic')

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx): 
        ann = self.ann[idx]

        data = {}
        data['id'] = ann['id']
        
        image = ann['image']
        data['image'] = self.loader(image).convert('RGB') if self.return_image else image

        if 'label' in ann:
            data['label'] = ann['label']

        if self.transform is not None and self.return_image:
            data['images_tensor'] = self.transform(data['image'])
            data['image_tensor_mask'] = np.zeros(data['images_tensor'][0].shape[-2:])[None,]

        if 'query' in ann:
            data['query'] = ann['query']

        if 'bbox' in ann:
            x1, y1, x2, y2 = ann['bbox']
            assert x1 <= x2 and y1 <= y2, ann
            
            data['bbox'] = (
                x1 / data['image'].width * self.box_scale,
                y1 / data['image'].height * self.box_scale,
                x2 / data['image'].width * self.box_scale,
                y2 / data['image'].height * self.box_scale,
            )
            factor=self.resolution/self.box_scale
            m_x1,m_y1,m_x2,m_y2=data['bbox'][0]*factor,data['bbox'][1]*factor,data['bbox'][2]*factor,data['bbox'][3]*factor
            data['image_tensor_mask'][:,int(m_y1):int(m_y2),int(m_x1):int(m_x2)]=1



        return self.data_augment(data)

    def shuffle(self):
        random.shuffle(self.ann)

    @staticmethod
    def allow_random_crop(caption):
        keywords = ['top', 'bottom', 'left', 'right', 'center', 'middle', 'above', 'below', 'first', 'second', 'third']
        for keyword in keywords:
            if keyword in caption:
                return False
        return True

    def data_augment(self, data):
        if self.random_flip and random.random() < 0.5:
            data['image'] = data['image'].transpose(Image.FLIP_LEFT_RIGHT)
            data['images_tensor'] = self.transform(data['image'])
            data['image_tensor_mask'] = np.zeros(data['images_tensor'][0].shape[-2:])[None,]
            
            caption = data['label']
            caption = caption.replace('left', '<LEFT>')
            caption = caption.replace('right', '<RIGHT>')
            # print(f'[caption befor flip] {data["label"]}')
            data['label'] = caption.replace('<LEFT>', 'right').replace('<RIGHT>', 'left')
            # print(f'[caption after flip] {data["label"]}')

            x1, y1, x2, y2 = data['bbox']
            x1 = x1 / self.box_scale
            y1 = y1 / self.box_scale
            x2 = x2 / self.box_scale
            y2 = y2 / self.box_scale

            flip_x1 = 1 - x1
            flip_x2 = 1 - x2
            x1 = flip_x2
            x2 = flip_x1

            data['bbox'] = (
                x1 * self.box_scale,
                y1 * self.box_scale,
                x2 * self.box_scale,
                y2 * self.box_scale,
            )

            factor=self.resolution/self.box_scale
            m_x1,m_y1,m_x2,m_y2=data['bbox'][0]*factor,data['bbox'][1]*factor,data['bbox'][2]*factor,data['bbox'][3]*factor
            data['image_tensor_mask'][:,int(m_y1):int(m_y2),int(m_x1):int(m_x2)]=1

        # cv2.imwrite('test.jpg',cv2.cvtColor(data['images_tensor'][0].transpose(1,2,0)*255, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('test_mask.jpg',np.repeat(data['image_tensor_mask'],3,axis=0).transpose(1,2,0)*255)
        # print(data['label'])
        if self.allow_random_crop(data['label']) and random.random() < self.random_resize_crop_prob:
            image = data['image']
            x1, y1, x2, y2 = data['bbox']
            bbox = (
                x1 / self.box_scale * image.width,
                y1 / self.box_scale * image.height,
                x2 / self.box_scale * image.width,
                y2 / self.box_scale * image.height,
            )
            
            image, bbox = self.random_resize_crop(image, bbox)
            data['image'] = image
            data['images_tensor'] = self.transform(data['image'])
            data['image_tensor_mask'] = np.zeros(data['images_tensor'][0].shape[-2:])[None,]
            
            x1, y1, x2, y2 = bbox
            data['image_tensor_mask'][:,int(y1):int(y2),int(x1):int(x2)]=1
            bbox = (
                x1 / self.resolution * self.box_scale,
                y1 / self.resolution * self.box_scale,
                x2 / self.resolution * self.box_scale,
                y2 / self.resolution * self.box_scale,
            )
            data['bbox'] = bbox
            # print(f'[caption after random_resize_crop] {data["label"]}')
        # cv2.imwrite('test.jpg',cv2.cvtColor(data['images_tensor'][0].transpose(1,2,0)*255, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('test_mask.jpg',np.repeat(data['image_tensor_mask'],3,axis=0).transpose(1,2,0)*255)
        # print(data['label'])
        x1, y1, x2, y2 = data['bbox']
        data['bbox'] = (int(x1), int(y1), int(x2), int(y2))

        return data
    
class RefCOCOEvalDataset(GroundingBaseEvalDataset):
    def __init__(
        self,
        data_root: str,
        annt_root: str,
        split:str = 'refcoco_testA',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.annt_file = os.path.join(annt_root,f'{split}.json')

        annotations = load_json(self.annt_file, self.loader.client)
        for ann in annotations:
            item = {
                'id': ann['img_id'],
                'image': os.path.join(data_root, '{}.jpg'.format(ann['img_id'][:27])),
                'label': ann['sents'],
            }

            if 'bbox' in ann:
                # x1y1x2y2
                x1,y1,w,h=ann['bbox']
                item['bbox'] = [int(x1),int(y1),int(x1+w),int(y1+h)]

            self.ann.append(item)
        
        # print(self.ann[:10])
        self.shuffle()
        self.ann=self.ann[:400]


class RandomResizedCrop(RandomResizedCropAndInterpolation):
    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        x1, y1, x2, y2 = bbox

        i = min(y1, i)
        j = min(x1, j)
        h = max(y2, i+h) - i
        w = max(x2, j+w) - j
        
        bbox = [x1-j, y1-i, x2-j, y2-i]
        bbox[0] = bbox[0] / w * self.size[0]
        bbox[1] = bbox[1] / h * self.size[1]
        bbox[2] = bbox[2] / w * self.size[0]
        bbox[3] = bbox[3] / h * self.size[1]
        
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation), tuple(bbox)