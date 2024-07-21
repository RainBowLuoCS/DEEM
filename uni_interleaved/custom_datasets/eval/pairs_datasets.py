import os
import json
import random
import torch
import numpy as np
from collections import Counter

from ..utils.loader import BaseDataset
from ..utils.wds_utils import init_tokenizer
from ..utils.re_rank import RICES


class ImageTextPairEvalCollator:
    def __init__(
        self,
        tokenizer_path,
        uncond_prob=0.0,
        num_img_token=32,
        img_first_prob=1.0,
        generation_kwargs=None,
        instr_prompts=None,
        padding="longest",
        train_dataset=None,
        few_shot_n_shot=2,
        few_shot_template="Caption: {caption} {image}",
        use_rice=True,
        rice_encoder="./assets/openai/clip-vit-large-patch14",
        cached_features_path=None,
    ):
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.num_img_token = num_img_token
        # NOTE the prob is used to control the mode of generation "generate text" 1.0 or  "generate image" 0.0
        self.img_first_prob = img_first_prob
        self.uncond_prob = uncond_prob

        default_generation_kwargs = dict(
            max_length=20,
            min_length=8,
            length_penalty=1.,
            num_beams=5,
            top_p=0.9,
        )

        self.generation_kwargs = generation_kwargs or default_generation_kwargs
        self.padding = padding

        default_instr_prompts = {
            "image": [
                "## ASSISTANT: ",
                "## USER: please reconstruct the complete image from the description and the image to be filled in {caption} \n",
                "You are a helpful assistant. \n\n"
            ],
            "text": [
                "## ASSISTANT: A photo of",
                "## USER: {image} Could you provide a short description of the image? \n",
                "You are a helpful assistant. \n\n",
            ],
        }
        # default_instr_prompts = {
        #     "image": [
        #         "",
        #         "{few_shot_example} {caption}",
        #         ""
        #     ],
        #     "text": [
        #         "a photo of",
        #         "{few_shot_example} {image}",
        #         "",
        #     ],
        # }
        self.instr_prompts = instr_prompts or default_instr_prompts

        self.use_rice = use_rice
        self.train_dataset = train_dataset
        self.few_shot_n_shot = few_shot_n_shot
        self.few_shot_template = few_shot_template

        if self.use_rice:
            self.rice = RICES(
                dataset=self.train_dataset,
                batch_size=32,
                vision_encoder_path=rice_encoder,
                cached_features_path=cached_features_path,
            )

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
        meta = []

        text_inputs_with_prompt_image_all = []

        assis_prompt, user_prompt, sys_prompt = self.instr_prompts["text"]

        use_few_shot = (
            "{few_shot_example}" in user_prompt and self.train_dataset is not None
        )

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor, caption, index = data

            meta.append((index, caption))

            assert isinstance(images_tensor, tuple), images_tensor

            images_tensor, images_tensor_dec = images_tensor
            images_tensor = torch.from_numpy(images_tensor)
            _images_tensor_all = [images_tensor]
            images_tensor_dec = torch.from_numpy(images_tensor_dec)
            _images_tensors_dec_all = [images_tensor_dec]

            _num_image_per_seq = 1

            if use_few_shot:
                few_shot_example, images = self.get_few_shot_samples(
                    query_image=images_tensor
                )
                text_input = user_prompt.format(
                    few_shot_example=few_shot_example,
                    image=self.image_subseq,
                )
                # few-shot images first, then question image
                _images_tensor_all = images[0] + _images_tensor_all
                _images_tensors_dec_all = images[1] + _images_tensors_dec_all

                _num_image_per_seq += len(images)
            else:
                text_input = user_prompt.format(image=self.image_subseq)

            text_input = f"{sys_prompt} {text_input} {assis_prompt}".strip()

            images_tensors_all.extend(_images_tensor_all)
            images_tensors_dec_all.extend(_images_tensors_dec_all)
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

    def _call_for_generate_images(self, data_list):
        images_tensors_all = []
        images_tensors_dec_all = []
        captions = []
        num_image_per_seq=[]
        meta = []

        assis_prompt, user_prompt, sys_prompt = self.instr_prompts["image"]

        use_few_shot = (
            "{few_shot_example}" in user_prompt and self.train_dataset is not None
        )

        for data in data_list:
            images_tensor, caption, index = data

            assert isinstance(images_tensor, tuple), images_tensor

            images_tensor, images_tensor_dec = images_tensor
            images_tensor = torch.from_numpy(images_tensor)
            _images_tensor_all = [images_tensor]
            images_tensor_dec = torch.from_numpy(images_tensor_dec)
            _image_tensors_dec_all = [images_tensor_dec]
            _num_image_per_seq = 1

            if use_few_shot:
                few_shot_example, images = self.get_few_shot_samples(
                    query_image=images_tensor
                )
                text_input = user_prompt.format(
                    few_shot_example=few_shot_example,
                    caption=caption,
                )
                # few-shot images first, then question image
                _images_tensor_all = images[0] + _images_tensor_all
                _image_tensors_dec_all = images[1] + _image_tensors_dec_all

                _num_image_per_seq += len(images)
            else:
                text_input=user_prompt.format(
                    caption=caption,
                )
            images_tensors_all.extend(_images_tensor_all)
            images_tensors_dec_all.extend(_image_tensors_dec_all)
            num_image_per_seq.append(_num_image_per_seq)

            # text= "" if np.random.random() < self.uncond_prob else caption

            meta.append((index, caption))
     
            text = (
                f"{sys_prompt} {text_input} {assis_prompt} {self.image_subseq}"
            ).strip()

            # print(text)

            text = text.replace("<|sniffer|> ", "<|sniffer|>").replace(
                " <|startofimage|>", "<|startofimage|>"
            )

            captions.append(text)

        images_tensors = torch.stack(images_tensors_all, dim=0)
        image_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)

        assert image_tensors_dec.shape[0] == images_tensors.shape[0]

        self.tokenizer.padding_side = "right"
        text_tensor = self.tokenizer(
            captions,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask
        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )

        # prepare negative prompt_ids only for inference
        negative_prompt_ids = None
        if self.uncond_prob > 0.0:
            negative_prompt = self.image_subseq
            negative_prompt_tensor = self.tokenizer(
                [negative_prompt],
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                padding=self.padding,
                return_tensors="pt",
                return_attention_mask=True,
            )
            negative_prompt_ids = negative_prompt_tensor.input_ids

        data = dict(
            image_tensors=images_tensors,
            image_tensors_mask=None,
            image_tensors_dec=image_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            negative_prompt_ids=negative_prompt_ids,
            loss_txt_weight=0.0,
            meta=meta,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

    def get_few_shot_samples(self, query_image=None):
        images, images_dec = [], []

        if self.use_rice:
            samples = self.rice.find(query_image, self.few_shot_n_shot)[0]
        else:
            idx = random.sample(
                list(range(len(self.train_dataset))), self.few_shot_n_shot
            )
            samples = [self.train_dataset[i] for i in idx]

        few_shot_caption_only = "{image}" not in self.few_shot_template

        few_shot_example = ""
        for image, caption, _ in samples:
            if few_shot_caption_only:
                few_shot_example += self.few_shot_template.format(
                    caption=caption,
                )
            else:
       
                images.append(
                    torch.from_numpy(image[0])
                    if isinstance(image[0], np.ndarray)
                    else image[0]
                )
                images_dec.append(
                    torch.from_numpy(image[1])
                    if isinstance(image[1], np.ndarray)
                    else image[1]
                )

                few_shot_example += self.few_shot_template.format(
                    image=self.image_subseq,
                    caption=caption,
                )

        return few_shot_example, (images, images_dec)

class NoCapsEvalDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_file,
        transform,
        image_only=False,
        total_length=None,
        collate_mode='generate_texts',
    ) -> None:
        super().__init__()
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.image_only = image_only
        self.annts = self.load_annotations(annt_file)
        self.annt_file = annt_file
        if self.image_only:
            self.dedeup_image()
        if total_length is not None:
            self.annts = self.annts[:total_length]
        print(f"length of the dataset is {len(self.annts)}")

    def load_annotations(self, annt_file):
        meta_info = json.load(open(annt_file, "r"))
        images = meta_info['images']
        annotations = meta_info['annotations']

        image_info = {}
        for image in images:
            image_info[image['id']] = image

        processed_annotations = []
        for ann in annotations:
            image_id = ann['image_id']
            file_name = image_info[image_id]['file_name']
            caption = ann['caption']

            processed_annotations.append({
                'image': file_name,
                'caption': caption,
                'image_id': image_id,
            })

        return processed_annotations

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = annt["image_id"]
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def __repr__(self) -> str:
        return "Nocaps Dataset"

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"]
        if isinstance(caption, list):  # TODO, random choose one caption from the image
            caption = random.choice(caption)
        caption = caption.lower()
        image_idx_int = item["image_id"]
        image_path = os.path.join(self.data_root, item["image"])

        try:
            image = self.loader(image_path).convert("RGB")
            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption, image_idx_int

class Flickr30KEvalDataset(NoCapsEvalDataset):
    def __repr__(self) -> str:
        return "Flickr30K Dataset"

class Image2ParagraphEvalDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        collate_mode="generate_texts",
        phase="val",
    ) -> None:
        super().__init__()
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.phase = phase
        self.image_only = image_only

        annt_file = os.path.join(annt_root, f"paragraphs_coco.json")
        with open(annt_file, "r") as rf:
            data = json.load(rf)
        annts = {d["image_id"]: d for d in data["annotations"]}

        split_file = os.path.join(annt_root, f"{phase}_split.json")
        with open(split_file, "r") as rf:
            split_idxs = set(json.load(rf))
        annts = [v for k, v in annts.items() if k in split_idxs]

        self.annts = annts
        self.annt_file = annt_file
        if total_length is not None:
            self.annts = self.annts[:total_length]
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

        image_idx_int = item["image_id"]
        image_subpaths = item["url"].split("/")[-1:]
        image_path = os.path.join(self.data_root,'VG_100K',*image_subpaths)

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption, image_idx_int

class CocoCaptionEvalDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        collate_mode="generate_texts",
        shuffle=False,
        rerank_by_clip=False,
        phase="test",
        year="2014",
    ) -> None:
        super().__init__()
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.phase = phase
        self.year = year
        self.image_only = image_only
        self.rerank_by_clip = rerank_by_clip

        annt_file = os.path.join(
            annt_root, "annotations", f"coco_karpathy_{phase}.json"
        )
        # annt_file = os.path.join(
        #     annt_root, "annotations", f"coco_{phase}{year}.json"
        # )
        self.annt_file = annt_file
        self.annts = json.load(open(annt_file, "r"))
        # self.annts = json.load(open(annt_file, "r"))['annotations']
        if self.image_only:
            self.dedeup_image()
        if shuffle:
            np.random.shuffle(self.annts)
        if total_length is not None:
            self.annts = self.annts[:total_length]
        print(f"length of the dataset is {len(self.annts)}")

    # def dedeup_image(self):
    #     annts = {}
    #     for annt in self.annts:
    #         image_idx = str(annt["image_id"]).zfill(12)
    #         if image_idx in annts:
    #             continue
    #         annts[image_idx] = annt
    #     self.annts = list(annts.values())

    # def image_id_to_path(self, image_id):
    #     # coco-2014
    #     image_idx = str(image_id).zfill(12)
    #     image_name = f"COCO_{self.phase}{self.year}_{image_idx}.jpg"
    #     image_path = os.path.join(
    #         self.data_root, f"{self.phase}{self.year}", image_name
    #     )
    #     return image_path
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
            f"MSCOCO-Caption Dataset year={self.year} phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        
        caption=item["caption"]
        if isinstance(caption, list):
            caption = random.choice(caption)

        caption = caption.lower()

        # image_idx = str(item["image_id"]).zfill(12)
        # image_name = f"COCO_{self.phase}{self.year}_{image_idx}.jpg"
        # image_path = os.path.join(
        #     self.data_root, f"{self.phase}{self.year}", image_name
        # )
        
        image_name = item["image"]
        image_path = os.path.join(self.data_root, f"{image_name}")
        image_idx = image_name.split("_")[-1][:-4]

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption, image_idx
    
class LNCOCOEvalDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        collate_mode="generate_images",
        phase="val",
    ) -> None:
        super().__init__()
        assert phase == "val" and collate_mode in ["generate_images"]
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.phase = phase
        self.image_only = image_only

        annt_file = os.path.join(annt_root, "coco_val_captions.jsonl")
        with open(annt_file, "r") as rf:
            data = rf.readlines()
        self.annts = [json.loads(s) for s in data]
        self.annt_file = annt_file
        if self.image_only:
            self.dedeup_image()
        if total_length is not None:
            if total_length <= len(self.annts):
                self.annts = self.annts[:total_length]
            else:
                # over sampling
                cnter_image = Counter([a["image_id"] for a in self.annts])
                annts_weight = [1./cnter_image[a["image_id"]] for a in self.annts]
                annts_weight = [w / sum(annts_weight) for w in annts_weight]
                annts_n = np.random.choice(self.annts, total_length - len(self.annts), p=annts_weight)
                self.annts += list(annts_n)
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
        # coco-2017
        return os.path.join(self.data_root, "val2017", f"{image_id:012d}.jpg")

    def __repr__(self) -> str:
        return (
            f"LNCOCO Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"]
        # caption = ["small bird with a pale yellow underside light brown crown and back gray tail and wing tips tip of tail feather bright yellow black eyes and black strip over eyes",
        #            "an armchair in the shape of an avocad",
        #            "an astronaut riding a horse X, where X ∈ {“in a photorealistic style”, “in the style of Pop Art”, “as a charcoal sketch”, “as a golden relief”",
        #            "panda mad scientist mixing sparkling chemicals, art station",
        #             "an espresso machine that makes coffee X, art station, where X ∈ {“in a warm scene”, “from human soul”}",
        #             "a futuristic city X, where X ∈ {“in a synthwave style”, “in vaporwave style”, “made of water”, “Beijing opera style”}",
        #             "robots meditating in a vipassana retrea",
        #             "A sculpture of a duck made of wood",
        #             "A couple of glasses are sitting on a table",
        #             "A squirrel is inside a giant bright shiny crystal ball in on the surface of blue ocean. There are few clouds in the sky.",
        #             "An art gallery displaying Monet paintings. The art gallery is flooded. Robots are going around the art gallery using paddle boards.",
        #             "Oil-on-canvas painting of a blue night sky with roiling energy. A fuzzy and bright yellow crescent moon shining at the top. Below the exploding yellow stars and radiating swirls of blue, a distant village sits quietly on the right. Connecting earth and sky is a flame-like cypress tree with curling and swaying branches on the left. A church spire rises as a beacon over rolling blue hills.",
        #             "a long wooden bench in front of a brick wall",
        #             "a hot air balloon landing in a corn field",
        #             "Downtown Beijing at sunrise. detailed ink wash.",
        #             "a beat-up truck at the base of the Great Pyramid",
        #             "a wooden deck overlooking a mountain valley",
        #            "A duck with a vibrant green head, a yellow bill, and a distinctive white ring around its neck. Its chest is a rich brown color, while the rest of its body features shades of gray and white. The tail feathers are black with some white accents. The duck is swimming in clear blue water, creating a serene and natural setting.",
        #             "A sculpture of a duck made of wood",
        #             "a beat-up truck in the desert"]
        # caption = caption.lower()
        caption=caption[-1]
        image_idx_int = int(item["image_id"])
        image_path = os.path.join(self.data_root, "val2017", f"{image_idx_int:012d}.jpg")

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption, image_idx_int

# class MSCOCOEvalDataset(BaseDataset):
#     def __init__(
#         self,
#         data_root,
#         annt_root,
#         transform,
#         image_only=False,
#         total_length=None,
#         collate_mode="generate_images",
#         shuffle=False,
#         rerank_by_clip=False,
#         phase="val",
#         year="2014",
#     ) -> None:
#         super().__init__()
#         self.collate_mode = collate_mode
#         self.transform = transform
#         self.data_root = data_root
#         self.annt_root = annt_root
#         self.phase = phase
#         self.year = year
#         self.image_only = image_only
#         self.rerank_by_clip = rerank_by_clip

#         annt_file = os.path.join(
#             annt_root, "annotations", f"captions_{phase}{year}.json"
#         )
#         self.annt_file = annt_file
#         self.annts = json.load(open(annt_file, "r"))["annotations"]
#         if self.image_only:
#             self.dedeup_image()
#         if shuffle:
#             np.random.shuffle(self.annts)
#         if total_length is not None:
#             self.annts = self.annts[:total_length]
#         print(f"length of the dataset is {len(self.annts)}")

#     def dedeup_image(self):
#         annts = {}
#         for annt in self.annts:
#             image_idx = str(annt["image_id"]).zfill(12)
#             if image_idx in annts:
#                 continue
#             annts[image_idx] = annt
#         self.annts = list(annts.values())

#     def image_id_to_path(self, image_id):
#         # coco-2014
#         image_idx = str(image_id).zfill(12)
#         image_name = f"COCO_{self.phase}{self.year}_{image_idx}.jpg"
#         image_path = os.path.join(
#             self.data_root, f"{self.phase}{self.year}", image_name
#         )
#         return image_path

#     def __repr__(self) -> str:
#         return (
#             f"MSCOCO-Caption Dataset year={self.year} phase={self.phase}\n"
#             f"annotation_root={self.annt_root} data_root={self.data_root}\n"
#             f"transform={self.transform}"
#         )

#     def __len__(self):
#         return len(self.annts)

#     def __getitem__(self, index):
#         item = self.annts[index]
#         caption = item["caption"].lower()

#         image_idx = str(item["image_id"]).zfill(12)
#         image_name = f"COCO_{self.phase}{self.year}_{image_idx}.jpg"
#         image_path = os.path.join(
#             self.data_root, f"{self.phase}{self.year}", image_name
#         )
#         try:
#             image = self.loader(image_path).convert("RGB")

#             image = self.transform(image)
#         except:
#             print(image_path)
#             index = random.randint(0, len(self) - 1)
#             return self.__getitem__(index)

#         return image, caption, item["image_id"]
    
class MSCOCOEvalDataset(CocoCaptionEvalDataset):
    def __repr__(self) -> str:
        return (
            f"MSCOCO-Caption Dataset year={self.year} phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )