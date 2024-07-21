### Modified from https://github.com/CircleRadon/Osprey/tree/main/osprey/datasets

import copy
import os
import random
import numpy as np
import torch
from matplotlib import path

import json
import re

import torch.nn.functional as F
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image

from uni_interleaved.custom_datasets.utils.loader import BaseDataset
from uni_interleaved.custom_datasets.utils.wds_utils import init_tokenizer

HUMAN="user"
GPT="asistant"
BEGIN_SIGNAL = "## "
END_SIGNAL = "\n"
    
DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give ablout the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represtented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]

WHY_QUESTIONS = [
    'why?',
    'why',
    "What's the rationale for your decision?",
    'What led you to that conclusion?',
    "What's the reasoning behind your opinion?",
    'Why do you believe that to be true?',
    'Can you explain the basis for your thinking?',
    'What factors influenced your perspective?',
    'How did you arrive at that perspective?',
    'What evidence supports your viewpoint?',
    'What makes you think that way?',
    "What's the logic behind your argument?",
    'Can you provide some context for your opinion?',
    "What's the basis for your assertion?",
    'Why do you hold that belief?',
    'What experiences have shaped your perspective?',
    'What assumptions underlie your reasoning?',
    "What's the foundation of your assertion?",
    "What's the source of your reasoning?",
    "What's the motivation behind your decision?",
    "What's the impetus for your belief?",
    "What's the driving force behind your conclusion?",
    'Why do you think that?',
    "What's your reasoning?",
    'What makes you say that?',
    'Why do you feel that way?',
    "What's the story behind that?",
    "What's your thought process?",
    "What's the deal with that?",
    "What's the logic behind it?",
    'Why do you believe that?',
    "What's the real deal here?",
    "What's the reason behind it?",
    "What's the thought process behind your decision?",
    "What's the rationale for your opinion?",
    'Why do you have that impression?',
    "What's the background to that?",
    "What's the evidence that supports your view?",
    "What's the explanation for that?"
]

Ref_WAY = [
    'There are <region> in the image,',
    'There are some regions <region>,',
    'Given <region>,',
    'Given <region> in the image,',
    '<region>,',
    'Several regions <region> are in the image,',
    '<region> in the given image,'
]

VGQUESTIONS =  [
    'Give me a short description of <region>.',
    'Can you give me a short description of <region>?',
    'Can you provide me with a short description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in few words?",
    'What can you tell me about the region indicated by <region> in the image in few words?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a concise description?",
    'Could you describe the region shown as <region> in the picture concisely?',
    'What can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a brief description of the region marked with <region> in the image.',
    'Can you give me a brief introduction of the region labeled as <region> in the picture?',
    "I'm interested in knowing the region represented by <region> in the photo. Can you describe it in several words?",
    'What is the region outlined by <region> in the picture like? Could you give me a streamlined description?',
    'Can you provide me with a brief description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in few words, please?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a simple description?",
    'Could you describe the region shown as <region> in the picture in several words?',
    'Please provide me with a simple description of the region marked with <region> in the image, please.',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in few words, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a simple and clear description?',
    'Please describe the region <region> in the image concisely.',
    'Can you offer a simple analysis of the region <region> in the image?',
    'Could tell me something about the region highlighted by <region> in the picture briefly?',
    'Can you share a simple rundown of the region denoted by <region> in the presented image?'
]


def _spaced_points(low, high,n):
    """ We want n points between low and high, but we don't want them to touch either side"""
    padding = (high-low)/(n*2)
    return np.linspace(low + padding, high-padding, num=n)

def make_mask(height, width, box, polygons_list):
    """
    Mask size: int about how big mask will be
    box: [x1, y1, x2, y2, conf.]
    polygons_list: List of polygons that go inside the box
    """
    mask = np.zeros((height, width), dtype=np.bool_)
    
    xy = np.meshgrid(_spaced_points(box[0], box[2], n=width),
                     _spaced_points(box[1], box[3], n=height)) 
    xy_flat = np.stack(xy, 2).reshape((-1, 2))

    for polygon in polygons_list:
        polygon_path = path.Path(polygon)
        mask |= polygon_path.contains_points(xy_flat).reshape((height, width))
    return mask.astype(np.float32)

class ReferringMaskTrainCollator:
    def __init__(
        self,
        tokenizer_path,
        train_dataset=None,
        num_img_token=77,
        ignore_soi_token_loss=False,
        ignore_bos2soi_token_loss=False,
        max_length=2048,
    ):
        
        self.tokenizer = init_tokenizer(tokenizer_path,add_grounding_special_tokens=True)
        # remove <s> token
        self.begin_length=len(self.tokenizer(BEGIN_SIGNAL).input_ids)-1
        self.human_length=len(self.tokenizer(HUMAN).input_ids)-1
        self.num_img_token = num_img_token
        self.train_dataset=train_dataset
        self.max_length=max_length

        self.ignore_soi_token_loss = ignore_soi_token_loss
        self.ignore_bos2soi_token_loss = ignore_bos2soi_token_loss

        self.image_subseq = "<|sniffer|>" * self.num_img_token
        self.image_subseq = "<|startofimage|>" + self.image_subseq
        self.header=f"You are a helpful assistant.\n\n"
        self.no_empty_data=None

    def _add_speaker_and_signal(self, header, source, get_conversation=True):
        """Add speaker and start/end signal on each round."""

        conversation = header
        chunk_lens=[len(self.tokenizer(header).input_ids)]
        lens_sum=chunk_lens[0]
        for idx,sentence in enumerate(source):
            from_str = sentence["from"]
            if from_str.lower() == "human":
                from_str = HUMAN
            elif from_str.lower() == "gpt":
                from_str = GPT
            else:
                from_str = 'unknown'
            sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                                sentence["value"].replace("<mask>",self.image_subseq) + END_SIGNAL)

            chunk_lens.append(len(self.tokenizer(sentence["value"]).input_ids))
            lens_sum+=chunk_lens[-1]
            if lens_sum>self.max_length:
                if sentence["from"].lower() == "gpt":
                    chunk_lens=chunk_lens[:-2]
                elif  sentence["from"].lower() == "human":
                    chunk_lens=chunk_lens[:-1]
                break

            if get_conversation and sentence["from"].lower() == "gpt":
                conversation += source[idx-1]["value"]
                conversation += sentence["value"]

        conversation += self.tokenizer.eos_token

        return conversation, chunk_lens, conversation.count('<|startofimage|>')

    def __call__(self, data_list):
        
        return self._call_for_generate_texts(data_list)

    def _call_for_generate_texts(self, data_list):

        images_tensors_all = []
        images_tensors_dec_all=[]
        images_tensors_mask_all=[]
        num_image_per_seq = []
        text_ids=[]
        gt_text_ids=[]

        for data in data_list:
            images_tensor = data['image']
            masks = data['masks']
            conversations=data['conversations']
            
            assert isinstance(images_tensor, tuple), images_tensor

            _num_image_per_seq=1+len(masks)
            images_tensor, images_tensor_dec = images_tensor

            images_tensor = torch.from_numpy(images_tensor)
            _images_tensor_all = [images_tensor]*_num_image_per_seq
            images_tensor_dec = torch.from_numpy(images_tensor_dec)
            _images_tensor_dec_all = [images_tensor_dec]*_num_image_per_seq

            _images_tensor_mask_all=[torch.ones(images_tensor.shape[-2:])[None,]]
            
            # (N,H,W)
            assert len(masks.shape)==3, masks.shape
                  
            # (N,1,H,W)
            masks=torch.from_numpy(masks)[:,None,]

            _images_tensor_mask_all+=[i for i in F.interpolate(masks,size=images_tensor.shape[-2:])]

            text_input,chunk_lens,image_remain_len = self._add_speaker_and_signal(self.header, conversations)

            _num_image_per_seq=image_remain_len
            _images_tensor_all=_images_tensor_all[:image_remain_len]
            _images_tensor_dec_all=_images_tensor_dec_all[:image_remain_len]
            _images_tensor_mask_all=_images_tensor_mask_all[:image_remain_len]

            input_ids=torch.tensor(self.tokenizer(text_input).input_ids, dtype=torch.long)
            target = copy.deepcopy(input_ids)


            # chunk_lens=[len(self.tokenizer(i).input_ids) for i in [self.header]+[s["value"] for s in conversations]]

            speakers = [sentence["from"] for sentence in conversations]

            cur_idx = chunk_lens[0]
            chunk_lens= chunk_lens[1:]
            target[:cur_idx] = -100
            for tokenized_len, speaker in zip(chunk_lens, speakers):
                # remove start of sequnce token idx '<s>'
                tokenized_len-=1
                if speaker == "human":
                    target[cur_idx+self.begin_length+self.human_length:cur_idx + tokenized_len] = -100
                cur_idx += tokenized_len

            text_ids.append(input_ids)
            gt_text_ids.append(target)

            images_tensors_all.extend(_images_tensor_all)
            images_tensors_dec_all.extend(_images_tensor_dec_all)
            images_tensors_mask_all.extend(_images_tensor_mask_all)
            num_image_per_seq.append(_num_image_per_seq)


        text_ids = torch.nn.utils.rnn.pad_sequence(
            text_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        text_ids=text_ids[:,:self.max_length]
        attn_mask=text_ids.ne(self.tokenizer.pad_token_id)

        gt_text_ids = torch.nn.utils.rnn.pad_sequence(gt_text_ids,
                                                 batch_first=True,
                                                 padding_value=-100)
        
        gt_text_ids=gt_text_ids[:,:self.max_length]

        if len(images_tensors_all)==0:
            return self.no_empty_data
        
        images_tensors = torch.stack(images_tensors_all, dim=0)
        images_tensors_mask = torch.stack(images_tensors_mask_all, dim=0)
        images_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)

        assert images_tensors_dec.shape[0] == images_tensors.shape[0]
        assert images_tensors_mask.shape[0] == images_tensors.shape[0]

        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )

        data = dict(
            image_tensors=images_tensors,
            num_image_per_seq=num_image_per_seq,
            image_tensors_mask=images_tensors_mask,
            image_tensors_dec=images_tensors_dec,
            text_ids=text_ids,
            attention_mask=attn_mask,
            gt_text_ids=gt_text_ids,
            loss_img_weight=0.0,
        )

        self.no_empty_data=data

        return data

class ReferringTrainDataset(BaseDataset):

    def __init__(self,
                 ann_file=None,
                 data_root=None,
                 transform=None,
                 max_gt_per_img=15,
                 ):
        self.transform=transform
        self.max_gt_per_img = max_gt_per_img
        self.data_root=data_root

        self.data_infos = self.load_annotations(ann_file)
        self.shuffle()
        # self.data_infos=self.data_infos[:200]
        super().__init__()

    def __len__(self):
        return len(self.data_infos)
    
    def shuffle(self):
        random.shuffle(self.data_infos)    

    def load_annotations(self, ann_file):

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            info['filename'] = info['file_name']
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue

            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
    
    def get_ann_info(self, idx):

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info
    
    def annToMask(self, mask_ann, h, w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_text(self, data_item):
        image = data_item['img']
        ori_labels = data_item['gt_labels']
        ori_masks = np.array(data_item['gt_masks'])

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        if len(shuffle_ids)!=1:
            ori_masks = ori_masks[shuffle_ids]
        ori_labels = [ori_labels[i] for i in shuffle_ids]

        data_dict = dict()

        data_dict['conversations'] = []

        # print("num:",len(ori_labels))

        for i in range(len(ori_labels)):
            question = '<region>'
            question = question.replace('<region>', '<mask>')
            if i == 0:
                question = self.begin_str + question
            answer = ori_labels[i]
            data_dict['conversations'].append(
                {'from': 'human', 'value': question})
            data_dict['conversations'].append({'from': 'gpt', 'value': answer})

        data_dict['image'] = image # without transform
        data_dict['masks'] = ori_masks
        return data_dict

    def read_process_image(self, img_path):

        image = self.loader(img_path).convert('RGB')
        
        return self.transform(image)
    
    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path =os.path.join(self.data_root, data_info['filename'])
        image = self.read_process_image(img_path)

        gt_masks = []
        gt_labels = []
        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            gt_masks.append(mask)

            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])

        data_item = dict(
            img = image,
            gt_masks = gt_masks,
            gt_labels = gt_labels
        )
        return data_item

    def __getitem__(self, idx):

        data_item = self.get_data_item(idx)
        data_dict = self.process_text(data_item=data_item)

        return data_dict

class ReferringRefCOCO(ReferringTrainDataset):

    def __init__(self,
                 ann_file=None,
                 data_root=None,
                 transform=None
                 ):

        super().__init__(ann_file, data_root,transform)

        self.begin_str = '<mask>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image, as well as its position within ' \
                         'the image and its basic attributes.'

    def load_annotations(self, ann_file):

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            # info['filename'] = info['file_name'].split('_')[-1]
            info['filename'] = info['file_name']
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue
            
            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
        
    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path =os.path.join(self.data_root, data_info['filename'])
        image = self.read_process_image(img_path)

        gt_masks = []
        gt_labels = []
        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            gt_masks.append(mask)
            
            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(data_info['caption'])

        data_item = dict(
            img = image,
            gt_masks = gt_masks,
            gt_labels = gt_labels
        )
        return data_item

class ReferringRefCOCOP(ReferringRefCOCO):
    def __init__(self,
                 ann_file=None,
                 data_root=None,
                 transform=None,
                 ):
        super().__init__(ann_file, data_root, transform)
        self.begin_str = '<mask>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image and its basic attibuts, you should not ' \
                         'give its position within the image.'   
        
class ReferringCOCODataset(ReferringTrainDataset):

    def __init__(self,
                 ann_file=None,
                 data_root=None,
                 transform=None
                 ):

        super().__init__(ann_file, data_root,transform)
        self.begin_str = '<mask>\nIn the conversation below, you simply answer the category name based on what you see ' \
                        'in the imagery inside a particular region. I will give you only one region each time.\n' 

class ReferringPartImagenet(ReferringTrainDataset):

    def __init__(self,
                 ann_file=None,
                 data_root=None,
                 transform=None
                 ):

        super().__init__(ann_file, data_root,transform)
        CAT_CLASSES = (
            'Bottle', 'Biped', 'Quadruped', 'Fish', 'Reptile', 'Bicycle', 'Bird', 'Car', 'Boat', 'Snake', 'Aeroplane'
        )

        SUB_CLASSES = (
            'Tier', 'Hand', 'Wing', 'Mouth', 'Tail', 'Side', 'Fin', 'Engine', 'Foot', 'Head', 'Body', 'Sail', 'Seat'
        )

        begin_str = '<mask>\nIn the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'

class ReferringPascalPart(ReferringTrainDataset):

    def __init__(self,
                 ann_file=None,
                 data_root=None,
                transform=None,
                 ):

        super().__init__(ann_file, data_root, transform)
        CAT_CLASSES = ('potted plant', 'aeroplane', 'cow', 'cat', 'bus', 'horse', 'car', 
                    'dog', 'bicycle', 'person', 'bird', 'bottle', 'sheep', 'motorbike')

        SUB_CLASSES = ('eye', 'window', 'cap', 'headlight', 'hand', 'mirror', 'arm', 'plant', 
                    'wheel', 'ear', 'pot', 'foot', 'leg', 'nose', 'body', 'horn', 'handlebar', 
                    'neck', 'license plate', 'paw', 'saddle', 'head', 'muzzle', 'tail', 'wing', 
                    'beak', 'hair', 'torso', 'door', 'mouth')

        begin_str = '<mask>\n In the conversation below, you simply answer the category and subcategory name based on what you see' \
                            'in the image inside a particular region. It maybe a subpart of an object. '\
                            'I will give you only one region each time. Your answer should in the format of '\
                            'category:subcategory. '
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        self.begin_str = begin_str + class_str + subclass_str + '.\n'

class ReferringConversationDataset(ReferringTrainDataset):
    def __init__(self,
                 ann_file=None,
                 data_root=None,
                 transform=None
                 ):
        self.begin_str = "<mask>\nThis provides an overview of the picture.\n"
        super().__init__(ann_file, data_root,transform)


    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue
            masks = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]
            # filename = ann['file_name']
            img_path = os.path.join(self.data_root, filename)
            region_num = len(ann['annotation'])
            if region_num>self.max_gt_per_img:
                continue
            h, w = ann['height'], ann['width']
            str_region = ""
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)
                if i>0:
                    str_region+= ','
                str_region += "<ref>region{}</ref>".format(str(i+1))+"<mask>"

            for i in range(len(ann['conversations'])//2):
                    
                if i==0:
                    if region_num==1:
                        mid_str = "There are 1 part region in the picture: "+str_region+'. '
                    else:
                        mid_str = "There are {} part regions in the picture: ".format(str(region_num))+str_region+'. '

                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    question = self.begin_str+mid_str+question
                    qa_s.append({'from': 'human', 'value': question+self.limit})         
                else:
                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    qa_s.append({'from': 'human', 'value': question+self.limit})         

                
                answer = ann['conversations'][i*2+1]['value']
                answer = answer.replace('<','').replace('>','')
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos

    def __getitem__(self, i):
        data_info = self.data_infos[i]
        img_path = data_info['img_path']
        height = data_info['height']
        width = data_info['width']
        masks_raw = data_info['masks']
        masks = []
        for mask_r in masks_raw:
            mask = self.annToMask(mask_r, height, width)
            masks.append(mask)
            
        masks = np.array(masks)
        qas = data_info['qas']

        image = self.read_process_image(img_path)
        data_dict=dict()        

        data_dict['image'] = image
        data_dict['conversations']=qas
        data_dict['masks'] = masks

        return data_dict

class OspreyPartLevel(ReferringConversationDataset):
    def __init__(self,
                 ann_file=None,
                 data_root=None,
                    transform=None,
                 ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(ann_file, data_root,transform)

class OspreyLVISPosNeg(ReferringConversationDataset):
    def __init__(self,
                 ann_file=None,
                 data_root=None,
                    transform=None,
                 ):
        
        super().__init__(ann_file, data_root,transform)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue
            masks = []
            qa_s = []
            filename = ann['file_name']
            img_path = os.path.join(self.data_root, filename)
            region_num = len(ann['annotation'])
            if region_num>self.max_gt_per_img:
                continue
            h, w = ann['height'], ann['width']

            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)
        
            for i in range(len(ann['conversations'])//2):
                    
                question = ann['conversations'][i*2]['value']
                question = re.sub(r'<region\d+>', '<mask>', question)
                if i==0:
                    question = self.begin_str+question
                qa_s.append({'from': 'human', 'value': question})         
             
                answer = ann['conversations'][i*2+1]['value']
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
            # print(qa_s)

        return data_infos      

class OspreyConversations(ReferringConversationDataset):
    def __init__(self,
                 ann_file=None,
                 data_root=None,
                transform=None,
                 ):
        self.limit = ""
        super().__init__(ann_file, data_root,transform)

class OspreyShortForm(ReferringConversationDataset):
     def __init__(self,
                 ann_file=None,
                 data_root=None,
                transform=None,
                 ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(ann_file, data_root,transform)

class OspreyDetailedDescription(ReferringConversationDataset):
    def __init__(self,
                 ann_file=None,
                 data_root=None,
                transform=None,
                 ):
        super().__init__(ann_file, data_root,transform)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file))

        for ann in ann_list:
            masks = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]
            # filename = ann['file_name']
            img_path = os.path.join(self.data_root, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']

            if region_num>self.max_gt_per_img:
                continue

            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)

                question = random.choice(DETAILED_QUESTIONS)
                question = question.replace('<region>', '<mask>')
                if i==0:
                    qa_s.append({'from': 'human', 'value': self.begin_str+question})         
                else:
                    qa_s.append({'from': 'human', 'value': question})     
            
                answer = re.findall(r"<.*>:\ (.*)", ann['description'][i])[0]
           
                qa_s.append({'from': 'gpt', 'value': answer})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos

class ReferringVGDATA(ReferringTrainDataset):
    def __init__(self,
                 ann_file=None,
                 data_root=None,
                 transform=None,
                 ):

        self.ann_file = ann_file
        self.data_root=data_root
        self.limit=" Answer the question using a short phrase."

        super().__init__(ann_file, data_root, transform)

        self.begin_str = """<mask>\nThis provides an overview of the picture.\n"""


    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path = os.path.join(self.data_root, data_info['filename'])
        image = self.read_process_image(img_path)

        gt_labels = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            
            gt_labels.append(ann['caption'])
            gt_masks_ann.append(mask)


        data_item = dict(
            img = image,
            gt_labels=gt_labels,
            gt_masks=gt_masks_ann
        )
        return data_item

    
    def process_text(self, data_item):
        image = data_item['img']
        ori_labels = data_item['gt_labels']
        ori_masks = np.array(data_item['gt_masks'])

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        ori_masks = ori_masks[shuffle_ids]
        ori_labels = [ori_labels[i] for i in shuffle_ids]

        data_dict = dict()

        data_dict['conversations'] = []

        for i in range(len(ori_labels)):
            question = random.choice(VGQUESTIONS).strip()
            question = question.replace('<region>', '<mask>')
            if i == 0:
                question = self.begin_str + question
            question += self.limit
            answer = ori_labels[i]
            data_dict['conversations'].append(
                {'from': 'human', 'value': question})
            data_dict['conversations'].append({'from': 'gpt', 'value': answer})

        data_dict['image'] = image
        data_dict['masks'] = ori_masks
        return data_dict

class ReferringVCRDataset(BaseDataset):
    def __init__(self,
                 ann_file=None,
                 data_root=None,
                 transform=None
                 ):
        super(ReferringVCRDataset, self).__init__()

        self.data_root = data_root
        self.transform=transform
        self.begin_str = """<mask>.\nThis provides an overview of the picture.\n"""
        self.data_infos = self.load_annotations(ann_file)

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
          ann_list = [json.loads(line) for line in f]
        data_infos = []

        import re

        def replace_numbers_with_tags(s, class_names):
            pattern = r'\b(\d+)\b'
            try:
                result = re.sub(pattern, lambda match: f'{class_names[int(match.group(1))]} at region{match.group(1)}', s)
            except:
                # contain number not for instance
                return None
            return result


        for ann in ann_list:

            metadata_fn_path = ann['metadata_fn']
            img_fn = ann['img_fn']
            img_path = os.path.join(self.data_root,img_fn)
            annotations = json.load(open(os.path.join(self.data_root, metadata_fn_path)))
            masks = annotations['segms']
            bboxes = np.array(annotations['boxes'])

            class_names = ann['objects']
            num_objects = len(class_names)
            ref_string = ''
            for i in range(num_objects):
                ref_string = ref_string +  f'<ref>region{i+1}</ref> <mask>' + ','
            ref_string = ref_string[:-1]
            ref_prefix = random.choice(Ref_WAY)

            begion_string = ref_prefix.replace('<region>', ref_string)
            qa_s = []

            q = ann['question_orig']
            q = replace_numbers_with_tags(q, class_names)
            a = ann['answer_orig']
            a = replace_numbers_with_tags(a, class_names)
            why = ann['rationale_orig']
            why = replace_numbers_with_tags(why, class_names)
            if (q is None) or (a is None) or (why) is None:
                continue


            qa_s.append({'from': 'human', 'value': begion_string + q})
            qa_s.append({'from': 'gpt', 'value': a})
            qa_s.append({'from': 'human', 'value': random.choice(WHY_QUESTIONS)})
            qa_s.append({'from': 'gpt', 'value': why})

            data_infos.append(dict(
                img_path = img_path,
                bboxes = bboxes,
                masks = masks,
                labels= class_names,
                qas = qa_s)
            )


        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, i):
        data_info = self.data_infos[i]

        img_path = data_info['img_path']
        masks = data_info['masks']
        bboxes = data_info['bboxes']

        qas = data_info['qas']

        image = self.loader(img_path).convert('RGB')
        data_dict=dict()
        w, h = image.size
        # TODO ablation this

        pred_masks = np.zeros((len(masks), h, w))

        for i,mask in enumerate(masks):

            int_box =  [round(box) for box in bboxes[i][:-1]]
            
            height_ = int(int_box[3]-int_box[1])
            width_ = int(int_box[2]-int_box[0])
            box_mask = make_mask(height_, width_, bboxes[i], mask)

            pred_masks[i, int_box[1]:int_box[3], int_box[0]:int_box[2]] = box_mask

        image = self.transform(image)

        qas = copy.deepcopy(qas)
        qas[0]['value'] = self.begin_str + qas[0]['value']

        data_dict['conversations']=qas
        data_dict['image'] = image
        data_dict['masks'] = pred_masks

        return data_dict
