import os
import json
import random
import numpy as np

import torch

from ..utils.loader import BaseDataset
from ..utils.wds_utils import init_tokenizer


CAPTIONQUESTIONS = [
            'Briefly describe this image.',
            'Provide a concise depiction of this image.',
            'Present a short description of this image.',
            'Summarize this image in a few words.',
            'A short image caption:',
            'A short image description:',
            'A photo of ',
            'An image that shows ',
            'Write a short description for the image. ',
            'Write a description for the photo.',
            'Provide a description of what is presented in the photo.',
            'Briefly describe the content of the image.',
            'Can you briefly explain what you see in the image?',
            'Could you use a few words to describe what you perceive in the photo?',
            'Please provide a short depiction of the picture.',
            'Using language, provide a short account of the image.',
            'Use a few words to illustrate what is happening in the picture.',
]

IMAGEQUESTIONS = [
"please reconstruct the complete image from the description and the image to be filled in",
"can you restore the complete image based on the description and the existing parts of the image",
"Could you recreate the full image from the provided image description and the incomplete image?",
"What would the full image look like, given the description and the parts of the image provided?",
"Based on your understanding of the image description and the available sections of the image, could you restore it?",
"Please complete the picture based on what's described and what's already there in the picture.",
"Can you give a reconstruction of the complete image based on the description and the available parts?",
"Could you combine the description with the existing elements of the image to create a full image?",
"Using the provided description and the parts of the image, can you reconstruct the full image?",
"Can you recreate the key aspects of the image based on the description and the partial image?",
"What would the full image look like, considering the described features and the incomplete image provided?",
"Please recreate the image using the description and the partial image as a guide.",
"Can you reconstruct the overall theme or concept captured in the image based on the description and the partial image?",
"How would you recreate the image's composition and focus based on the provided description and image fragments?",
"What would the full image look like, considering the focal point or main subject described and the parts of the image available?",
"Considering the interactions of the different components described, how would you restore the full image?",
"Based on the fitting caption provided, how would you recreate the full image from the existing parts?",
"Can you create a complete image that captures the essence of the description and fills in the incomplete parts?",
"How would you recreate the full image based on the description summarizing the content in a phrase or sentence?",
"Please provide a reconstruction of the complete image based on the catchy and relevant caption and the partial image.",
"If you were to give this image a title, how would you complete the picture based on it and the available parts?",
"Considering the creative sentence describing the image, can you reconstruct the complete image?",
"Please suggest a way to restore the complete image based on the memorable phrase encapsulating the image's content.",
"What would the complete image look like based on the engaging phrase provided and the incomplete image?",
"Can you create a full image that highlights the main theme based on the description and the existing parts of the image?",
"How would you recreate the complete image to match the caption summarizing the image's story?",
"Provide a reconstruction of the complete image that conveys the core message described in the catchy caption.",
"If you were to give this image a headline, how would you reconstruct the full image from it and the available parts?",
"Can you craft a full image that communicates the essence based on the description and the partial image?",
"How would you reconstruct the complete image based on the powerful caption describing the image's content?",
"Please provide a restoration of the full image based on the inventive title summarizing the scene depicted in the partial image.",
"Compose a full image that reflects the key elements described in the concise and striking phrase.",
"If you were to create a caption for this image, how would you complete the picture based on it and the available parts?",
"Offer a complete image that highlights the central focus described in the compelling caption.",
"Can you produce a full image that encapsulates the overall mood described and fills in the incomplete parts?",
"Please generate a full image that would best illustrate the events captured in the description and the partial image",
"How would you express the main idea of the full image based on the impactful sentence and the partial image?",
"Please create a complete picture that conveys the essence of the description and the existing parts of the picture.",
"Compose a full image that reflects the most striking features described in the imaginative caption.",
"What would the full image look like based on the memorable statement representing the scene illustrated and the incomplete image?",
"Draft a complete image that brings the description to life for the viewer.",
"Can you suggest a full image that highlights the underlying message described and fills in the incomplete parts?",
"What would the complete image look like based on the engaging phrase conveying the action or subject matter depicted and the partial image?",
"How would you encapsulate the core theme of the full image in a concise and expressive manner based on the description and the incomplete image?",
"Please provide a full image that captures the spirit of the description and the partial image.",
"Craft a complete image that showcases the most prominent attributes described in the captivating caption.",
"What would the complete image look like based on the intriguing statement summing up the scene presented and the partial image?",
"Develop a full image that paints a vivid picture for the viewer based on the descriptive caption and the available parts of the image.",
"Can you give a detailed account of what the full image would look like based on the image's content description?",
"What would the complete image look like considering the key elements and features described and the parts of the image visible?",
"How would you recreate the events or actions depicted in the full picture based on the narration and the incomplete picture?",
"Please share your reconstruction of the full image considering the various components described and present in the partial image.",
"What would the complete image look like, considering the overall theme or concept captured in the description and the parts of the image available? Can you create it?"
]

class ImageTextPairTrainCollator:
    def __init__(
        self,
        tokenizer_path,
        train_dataset=None,
        uncond_prob=0.0,
        num_img_token=77,
        img_first_prob=1.0,
        padding="longest",
   
    ):
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.num_img_token = num_img_token
        self.img_first_prob = img_first_prob
        self.uncond_prob = uncond_prob
        self.padding = padding

        self.train_dataset = train_dataset

        self.system_prompt="You are a helpful assistant.\n\n"

        self.caption_user_prompt="## USER: {image}{question} \n"
        self.image_user_prompt="## USER: {question}{caption} \n"

        self.caption_assist_prompt="## ASSISTANT:  A photo of"
        self.image_assist_prompt="## ASSISTANT: "

        self.image_subseq = "<|sniffer|>" * self.num_img_token

        self.image_subseq = "<|startofimage|>" + self.image_subseq


    def __call__(self, data_list):

        return self._call_for_train(data_list)

    def _call_for_train(self, data_list):
        if np.random.random() < self.img_first_prob:
            # image to text
            return self._call_for_generate_texts(data_list)
        else:
            # text to image
            return self._call_for_generate_images(data_list)
        
    def _call_for_generate_texts(self, data_list):
        images_tensors_all = []
        num_image_per_seq = []
        images_tensors_dec_all = []

        text_inputs_with_prompt_image_all = []

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor, caption= data

            assert isinstance(images_tensor, tuple), images_tensor

            images_tensor, images_tensor_dec = images_tensor
            images_tensor = torch.from_numpy(images_tensor)
            _images_tensor_all = [images_tensor]
            images_tensor_dec = torch.from_numpy(images_tensor_dec)
            _images_tensor_dec_all = [images_tensor_dec]

            _num_image_per_seq = 1

            text_input = self.caption_user_prompt.format(image=self.image_subseq,question=random.choice(CAPTIONQUESTIONS))

            text_input = f"{self.system_prompt} {text_input} {self.caption_assist_prompt}".strip()

            images_tensors_all.extend(_images_tensor_all)
            images_tensors_dec_all.extend(_images_tensor_dec_all)
            num_image_per_seq.append(_num_image_per_seq)


            ignore_prompt_token_offset = self.tokenizer(
                text_input.strip(), return_tensors="pt"
            ).attention_mask.sum(1)
            ignore_prompt_token_offsets.append(ignore_prompt_token_offset)
            text_input += " " + caption + self.tokenizer.eos_token

            text_inputs_with_prompt_image_all.append(text_input)

        self.tokenizer.padding_side = "right"
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
            meta={'dataset_name': self.train_dataset.dataset_name},
        )

        return data

    def _call_for_generate_images(self, data_list):

        images_tensors_all = []
        num_image_per_seq = []
        images_tensors_dec_all = []

        text_inputs_with_prompt_image_all = []

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor, caption= data

            assert isinstance(images_tensor, tuple), images_tensor

            images_tensor, images_tensor_dec = images_tensor
            images_tensor = torch.from_numpy(images_tensor)
            _images_tensor_all = [images_tensor]
            images_tensor_dec = torch.from_numpy(images_tensor_dec)
            _images_tensor_dec_all = [images_tensor_dec]

            _num_image_per_seq = 1

            text= "" if np.random.random() < self.uncond_prob else caption

            text_input = self.image_user_prompt.format(caption=text,question=random.choice(IMAGEQUESTIONS)) 

            text_input = f"{self.system_prompt} {text_input} {self.image_assist_prompt}".strip()

            images_tensors_all.extend(_images_tensor_all)
            images_tensors_dec_all.extend(_images_tensor_dec_all)
            num_image_per_seq.append(_num_image_per_seq)


            ignore_prompt_token_offset = 0
            ignore_prompt_token_offsets.append(ignore_prompt_token_offset)

            text_input += " " + self.image_subseq + self.tokenizer.eos_token

            text_inputs_with_prompt_image_all.append(text_input)

        self.tokenizer.padding_side = "right"
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
            loss_txt_weight=0.0,
            ignore_prompt_token_offset=ignore_prompt_token_offsets,
            meta={'dataset_name': self.train_dataset.dataset_name},
        )

        return data
        
class MSCOCOTrainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        shuffle=True,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.phase ="train"
        self.year = "2014"
        self.image_only = image_only

        annt_file = os.path.join(
            annt_root, "annotations", f"captions_{self.phase}{self.year}.json"
        )
        self.annt_file = annt_file
        self.annts = json.load(open(annt_file, "r"))["annotations"]

        if shuffle:
            np.random.shuffle(self.annts)

        if self.image_only:
            self.dedeup_image()

        if total_length is not None:
            self.annts = self.annts[:total_length]
        print(f"length of the dataset is {len(self.annts)}")

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = str(annt["image_id"]).zfill(12)
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def __repr__(self) -> str:
        return (
            f"MSCOCO-Caption Dataset year={self.year} phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"].lower()

        image_idx = str(item["image_id"]).zfill(12)
        image_name = f"COCO_{self.phase}{self.year}_{image_idx}.jpg"
        image_path = os.path.join(
            self.data_root, f"{self.phase}{self.year}", image_name
        )
        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption

class CocoCaptionKarpathyTrainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        shuffle=True,
        total_length=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        phase="train"
        year="2014"
        self.phase = phase
        self.year = year
        self.image_only = image_only
        annt_file = os.path.join(
            annt_root, "annotations", f"coco_karpathy_{phase}.json"
        )
        self.annts = json.load(open(annt_file, "r"))
        self.annt_file = annt_file
        if self.image_only:
            self.dedeup_image()
        if total_length is not None:
            self.annts = self.annts[:total_length]

        if shuffle:
            np.random.shuffle(self.annts)

        print(f"length of the dataset is {len(self.annts)}")

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = annt["image"].split("_")[-1][
                :-4
            ]  # 'val2014/COCO_val2014_000000391895.jpg'
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def image_id_to_path(self, image_id):
        phase = "val" if self.phase == "test" else self.phase
        # coco-2014
        image_idx = str(image_id).zfill(12)
        image_name = f"COCO_{phase}{self.year}_{image_idx}.jpg"
        image_path = os.path.join(
            self.data_root, f"{phase}{self.year}", image_name
        )
        return image_path

    def __repr__(self) -> str:
        return (
            f"MSCOCO-Caption Karpathy Dataset year={self.year} phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"]
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = caption.lower()

        image_name = item["image"]
        image_path = os.path.join(self.data_root, f"{image_name}")

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption

class LNCOCOTrainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        shuffle=True,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        phase="train"
        year="2017"
        self.phase = phase
        self.year = year
        self.image_only = image_only
        annt_file = os.path.join(
            annt_root, f"coco_{phase}_captions.jsonl"
        )
        self.annts =[]
        with open(annt_file, "r") as f:
            lines=f.readlines()
        for line in lines:
            self.annts.append(json.loads(line))

        self.annt_file = annt_file
        if self.image_only:
            self.dedeup_image()
        if total_length is not None:
            self.annts = self.annts[:total_length]
        if shuffle:
            np.random.shuffle(self.annts)
        print(f"length of the dataset is {len(self.annts)}")

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = annt["image_id"]
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def image_id_to_path(self, image_id):
        # coco-2014
        image_idx = str(image_id).zfill(12)
        image_name = f"{image_idx}.jpg"
        image_path = os.path.join(
            self.data_root, f"{self.phase}{self.year}", image_name
        )
        return image_path

    def __repr__(self) -> str:
        return (
            f"LNCOCO-Caption Dataset year={self.year} phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"]
        caption = caption.lower()

        image_idx = item['image_id'].zfill(12)
        image_name = f"{image_idx}.jpg"
        
        image_path = os.path.join(self.data_root,f"{self.phase}{self.year}",f"{image_name}")

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption

class TextCapsTrainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        shuffle=True
    ) -> None:
        super().__init__()
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root

        self.image_only = image_only
        annt_file = os.path.join(
            annt_root, f"TextCaps_0.1_train.json"
        )
        with open(annt_file, 'r') as f:
            self.annts = json.load(f)['data']

        self.annt_file = annt_file
        if self.image_only:
            self.dedeup_image()
        if total_length is not None:
            self.annts = self.annts[:total_length]
        if shuffle:
            np.random.shuffle(self.annts)
        print(f"length of the dataset is {len(self.annts)}")

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = annt["image_id"]
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def __repr__(self) -> str:
        return (
            f"TextCaps Dataset \n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption_str"]
        caption = caption.lower()

        image_file = '{}.jpg'.format(item['image_id'])

        image_path = os.path.join(self.data_root, image_file)


        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption

class Flickr30kCaptionTrainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        shuffle=True
    ) -> None:
        super().__init__()
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root

        self.image_only = image_only

        annt_file = os.path.join(annt_root, "groundedcaption.json")
        with open(annt_file, "r") as rf:
            data = json.load(rf)
        annts = {d["image_id"]: d for d in data}

        split_file = os.path.join(annt_root, "flickr30k_test1k.json")
        with open(split_file, "r") as rf:
            split_idxs = set([i['file_name'].split('.')[0] for i in json.load(rf)['images']])
        self.annts = [v for k, v in annts.items() if k not in split_idxs]

        self.annt_file = annt_file
        if self.image_only:
            self.dedeup_image()
        if total_length is not None:
            self.annts = self.annts[:total_length]
        if shuffle:
            np.random.shuffle(self.annts)
        print(f"length of the dataset is {len(self.annts)}")

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = annt["image_id"]
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def __repr__(self) -> str:
        return (
            f"Flickr30k Dataset \n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["detailed_caption"]
        caption = caption.lower()

        image_file = '{}.jpg'.format(item['image_id'])

        image_path = os.path.join(self.data_root, image_file)


        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption

class Image2ParagraphTrainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        shuffle=True
    ) -> None:
        super().__init__()
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        phase="train"
        self.phase = phase
        self.image_only = image_only

        annt_file = os.path.join(annt_root, "paragraphs_coco.json")
        with open(annt_file, "r") as rf:
            data = json.load(rf)
        annts = {d["image_id"]: d for d in data["annotations"]}

        split_file = os.path.join(annt_root,  f"{phase}_split.json")
        with open(split_file, "r") as rf:
            split_idxs = set(json.load(rf))
        annts = [v for k, v in annts.items() if k in split_idxs]

        self.annts = annts
        self.annt_file = annt_file
        if total_length is not None:
            self.annts = self.annts[:total_length]
        if shuffle:
            np.random.shuffle(self.annts)
        print(f"length of the dataset is {len(self.annts)}")

    def __repr__(self) -> str:
        return (
            f"Image2Paragraph Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"]
        # caption = caption.lower()
        image_subpaths = item["url"].split("/")[-1:]
        image_path = os.path.join(self.data_root,'VG_100K',*image_subpaths)

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption
