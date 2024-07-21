import re
import os
import json
import random
import torch
import pickle
import time
import itertools
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
import torchvision.transforms.functional as F
from timm.data.transforms import RandomResizedCropAndInterpolation
from visual_genome import local
from ..utils.wds_utils import init_tokenizer
from ..utils.loader import BaseDataset
from torch.utils.data import IterableDataset


class REFER:
    def __init__(self, data_root, vis_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        self.ann_dir = os.path.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.vis_root = vis_root
        elif dataset == 'refclef':
            raise 'No RefClef image data'
        else:
            raise 'No refer dataset is called [%s]' % dataset

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = os.path.join(self.ann_dir, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = pickle.load(open(ref_file, 'rb'))

        # load annotations from data/dataset/instances.json
        instances_file = os.path.join(self.ann_dir, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if
                            split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    raise 'No such split [%s]' % split
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='box'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(os.path.join(self.vis_root, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                raise NotImplementedError('RefClef is not downloaded')
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

class GroundingTrainCollator:
    def __init__(
        self,
        tokenizer_path,
        train_dataset=None,
        num_img_token=77,
        ignore_soi_token_loss=False,
        ignore_bos2soi_token_loss=False,
        max_length=2048,
    ):
        # self.tasks=("grounding", "referring")
        # self.tasks=("grounding",)
        self.tasks=("referring",)
        
        self.tokenizer = init_tokenizer(tokenizer_path,add_grounding_special_tokens=True)
        self.num_img_token = num_img_token
        self.train_dataset=train_dataset
        self.max_length=max_length

        self.ignore_soi_token_loss = ignore_soi_token_loss
        self.ignore_bos2soi_token_loss = ignore_bos2soi_token_loss

        self.tasks_prompts={
            "grounding":[
                "## ASSISTANT: the bounding box coordinate is ",
                "## USER: {image} Provide the bounding box coordinate of the region this sentence describes: {caption}. \n",
                "You are a helpful assistant.\n\n",
            ],
             "referring":
            [
                "## ASSISTANT:",
                "## USER: {image} \n I will provide you with only one region containing only one object, although there may be other objects present in the image. It is recommended that you describe the object's relative position with respect to other objects in the image and its basic attibuts, you should not give its position within the image. {mask}. \n",
                "You are a helpful assistant.\n\n",
            ]
        }

        self.image_subseq = "<|sniffer|>" * self.num_img_token
        self.image_subseq = "<|startofimage|>" + self.image_subseq

    def box2str(self, box):
        x1, y1, x2, y2 = box
        assert x1 <= x2 and y1 <= y2
        return f"({x1:03d},{y1:03d})({x2:03d},{y2:03d})"


    def __call__(self, data_list):
        concat_mode = [data.get('concat_mode', False) for data in data_list]
        assert all(concat_mode) or not any(concat_mode)
        
        if all(concat_mode):
            return self._call_for_concat_mode(data_list)
        
        return self._call_for_generate_texts(data_list)

    def _call_for_generate_texts(self, data_list):

        images_tensors_all = []
        images_tensors_dec_all=[]
        images_tensors_mask_all=[]
        num_image_per_seq = []
        text_inputs_with_prompt_image_all = []

        task=random.choice(self.tasks)
        assis_prompt, user_prompt, sys_prompt=self.tasks_prompts[task]

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor = data['images_tensor']
            answer = data['label']
            
            assert isinstance(images_tensor, tuple), images_tensor

            images_tensor, images_tensor_dec = images_tensor

            images_tensor = torch.from_numpy(images_tensor)
            images_tensor_dec = torch.from_numpy(images_tensor_dec)

            _images_tensor_all = [images_tensor]
            _images_tensor_mask_all = [torch.ones(images_tensor.shape[-2:])[None,]]
            _images_tensor_dec_all = [images_tensor_dec]
            _num_image_per_seq=1

            # NOTE add object once again
            if task=='referring':
                _images_tensor_mask_all.append(torch.from_numpy(data['image_tensor_mask']))
                _images_tensor_all.append(images_tensor)
                _images_tensor_dec_all.append(images_tensor_dec)
                _num_image_per_seq+=1

            if task == 'grounding':
                box = self.box2str(data['bbox'])
                text_input = user_prompt.format(
                    image=self.image_subseq, caption=answer,
                )
            elif task == "referring":
                text_input = user_prompt.format(
                    image=self.image_subseq, mask=self.image_subseq,
                )
            else:
                raise NotImplementedError()

            text_input = f"{sys_prompt} {text_input} {assis_prompt}".strip()

            images_tensors_all.extend(_images_tensor_all)
            images_tensors_dec_all.extend(_images_tensor_dec_all)
            images_tensors_mask_all.extend(_images_tensor_mask_all)
            num_image_per_seq.append(_num_image_per_seq)


            ignore_prompt_token_offset = self.tokenizer(
                text_input.strip(), return_tensors="pt"
            ).attention_mask.sum(1)
            ignore_prompt_token_offsets.append(ignore_prompt_token_offset)
            
            if task == "grounding":
                box = self.box2str(data['bbox'])
                # text_input += f" <box>{box}</box>{self.tokenizer.eos_token}"
                text_input += f"<box>{box}</box>{self.tokenizer.eos_token}"
            else:
                text_input += " " + answer + self.tokenizer.eos_token
            # print(text_input)
            text_inputs_with_prompt_image_all.append(text_input)

        self.tokenizer.padding_side = "right"
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

        # Modified from _prepare_gt_text_ids()
        gt_text_ids = text_ids.clone()
        assert gt_text_ids.shape[0] == len(ignore_prompt_token_offsets), f'{gt_text_ids.shape[0]},{len(ignore_prompt_token_offsets)}'

        for idx, offset in enumerate(ignore_prompt_token_offsets):
            gt_text_ids[idx, :offset] = -100

        gt_text_ids = gt_text_ids.masked_fill(
            text_ids == self.tokenizer.pad_token_id, -100
        )
        gt_text_ids = gt_text_ids.masked_fill(
            text_ids == self.tokenizer.convert_tokens_to_ids('<|sniffer|>'), -100
        )
        gt_text_ids = gt_text_ids.masked_fill(attn_mask == 0, -100)
        if self.ignore_bos2soi_token_loss:
            is_bos_token = text_ids[:-1] == self.tokenizer.convert_tokens_to_ids('<s>')
            is_soi_token = text_ids[1:] == self.tokenizer.convert_tokens_to_ids('<|startofimage|>')
            is_bos2soi_token = torch.logical_and(is_bos_token, is_soi_token)
            gt_text_ids[1:] = gt_text_ids[1:].masked_fill(is_bos2soi_token, -100)
        if self.ignore_soi_token_loss:
            gt_text_ids = gt_text_ids.masked_fill(
                text_ids == self.tokenizer.convert_tokens_to_ids('<|startofimage|>'), -100
            )

        gt_text_ids = gt_text_ids.contiguous()


        data = dict(
            image_tensors=images_tensors,
            num_image_per_seq=num_image_per_seq,
            image_tensors_mask=images_tensors_mask,
            image_tensors_dec=images_tensors_dec,
            text_ids=text_ids,
            attention_mask=attn_mask,
            gt_text_ids=gt_text_ids,
            loss_img_weight=0.0,
            ignore_prompt_token_offset=ignore_prompt_token_offsets,
            meta={'dataset_name': self.train_dataset.dataset_name},
        )

        return data
    
    def _call_for_concat_mode(self, data_list):
        image_tensors = []
        image_tensors_mask=[]
        image_tensors_dec=[]
        num_image_per_seq = []
        text_ids = []
        attn_mask = []
        gt_text_ids = []

        for data in data_list:
            image_tensors.append(data['image_tensors'])
            image_tensors_mask.append(data['image_tensors_mask'])
            image_tensors_dec.append(data['image_tensors_dec'])
            num_image_per_seq.append(data['num_image_per_seq'])
            
            assert data['text_ids'].shape[0] == 1
            assert data['attention_mask'].shape[0] == 1
            assert data['gt_text_ids'].shape[0] == 1
            
            text_ids.append(data['text_ids'].squeeze(0))
            attn_mask.append(data['attention_mask'].squeeze(0))
            gt_text_ids.append(data['gt_text_ids'].squeeze(0))

        image_tensors = torch.cat(image_tensors)
        image_tensors_dec = torch.cat(image_tensors_dec)
        image_tensors_mask = torch.cat(image_tensors_mask)
        num_image_per_seq = torch.stack(num_image_per_seq)
        text_ids = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gt_text_ids = torch.nn.utils.rnn.pad_sequence(gt_text_ids, batch_first=True, padding_value=-100)

        data = dict(
            image_tensors=image_tensors,
            image_tensors_dec=image_tensors_dec,
            image_tensors_mask=image_tensors_mask,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            gt_text_ids=gt_text_ids,
            loss_img_weight=0.0,
        )

        return data
    
class GroundingBaseTrainDataset(BaseDataset):
    def __init__(
        self,
        transform = None,
        box_scale = 999,
        random_flip: bool = False,
        random_resize_crop_prob: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ann = []
        self.box_scale = box_scale
        self.transform = transform
        self.resolution=self.transform.transform1.resolution

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
        data['image'] = self.loader(image).convert('RGB')

        if 'label' in ann:
            data['label'] = ann['label']

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

        x1, y1, x2, y2 = data['bbox']
        data['bbox'] = (int(x1), int(y1), int(x2), int(y2))

        return data

class RefCOCOTrainDataset(GroundingBaseTrainDataset):
    def __init__(
        self,
        data_root: str,
        annt_root: str,
        data_type: str = 'refcoco',
        split_type: str ='unc',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.annt_root = annt_root
        self.refer = REFER(annt_root, data_root, data_type, split_type)
        self.ref_ids = self.refer.getRefIds(split="train")
        for ref_id in self.ref_ids:
            ref = self.refer.loadRefs(ref_id)[0]

            image_file = 'COCO_train2014_{:0>12}.jpg'.format(ref["image_id"])
            image_path = os.path.join(self.data_root, image_file)

            sample_sentence = random.choice(ref['sentences'])['raw']

            bbox = self.refer.getRefBox(ref['ref_id'])
            bbox = [
                bbox[0],
                bbox[1],
                (bbox[0] + bbox[2]),
                (bbox[1] + bbox[3])
            ]
            bbox = [int(x) for x in bbox]
            self.ann.append({
                "id":-1,
                "image": image_path,
                "label": sample_sentence,
                "bbox": bbox
            })

        self.shuffle()

class VisualGenomeTrainDataset(GroundingBaseTrainDataset):
    def __init__(
        self,
        data_root: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        all_regions = local.get_all_region_descriptions(data_root)
        all_regions = [region for regions in all_regions for region in regions]
        regions = [region for region in all_regions if region.width * region.height < 16384]

        for region in regions:

            image_file = region.image.url.split('/')[-1:]
            bbox=[region.x, region.y, region.x+region.width, region.y+region.height]
            bbox = [int(x) for x in bbox]
            self.ann.append({
                "id":-1,
                "image": os.path.join(self.data_root,'VG_100K',*image_file),
                "label": region.phrase,
                "bbox": bbox
            })

        self.shuffle()    

# class Filcrk30KTrainDataset(GroundingBaseTrainDataset):
#     def __init__(
#         self,
#         data_root: str,
#         annt_root: str,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.data_root = data_root
#         anns = json.load(open(os.path.join(annt_root,"phrasetobbox.json"), 'r') )

#         for info in anns:
#             image_file = '{}.jpg'.format(info['image_id'])
#             image_path = os.path.join(self.data_root, image_file)

#             self.ann.append({
#                 "id":-1,
#                 "image": image_path,
#                 "label": info["phrase"],
#                 "bbox": info["bbox"]
#             })

#         self.shuffle()  
        

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