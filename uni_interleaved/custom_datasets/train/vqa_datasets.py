import os
import json
import random
import torch
import numpy as np
from ..utils.loader import BaseDataset
from ..utils.wds_utils import init_tokenizer

QUESTIONS = [
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

# This class is designed to conduct stf training process on vqa and caption dataset
class VQACaptionTrainCollator:
    def __init__(
        self,
        tokenizer_path,
        num_img_token=77,
        train_dataset=None,
    ):
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.train_dataset = train_dataset

        self.num_img_token = num_img_token

        self.system_prompt="You are a helpful assistant.\n\n"
        self.user_prompt="## USER: Based on the image, please answer the question. {image}{question}\n"
        self.assist_prompt="## ASSISTANT:  The answer is"

        self.image_subseq = "<|sniffer|>" * self.num_img_token
        self.image_subseq = "<|startofimage|>" + self.image_subseq

    def __call__(self, data_list):

        images_tensors_all = []
        num_image_per_seq = []
        images_tensors_dec_all = []
        text_inputs_with_prompt_image_all = []

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor, question, answer= data

            assert isinstance(images_tensor, tuple), images_tensor

            images_tensor, images_tensor_dec = images_tensor

            images_tensor = torch.from_numpy(images_tensor)
            images_tensor_dec = torch.from_numpy(images_tensor_dec)
            images_tensors_dec_all.append(images_tensor_dec)
            images_tensors_all.append(images_tensor)

            num_image_per_seq.append(1)

            text_input = self.user_prompt.format(
                    image=self.image_subseq, question=question
                )
            
            text_input = f"{self.system_prompt} {text_input} {self.assist_prompt}".strip()

            ignore_prompt_token_offset = self.tokenizer(
                text_input.strip(), return_tensors="pt"
            ).attention_mask.sum(1)

            ignore_prompt_token_offsets.append(ignore_prompt_token_offset)

            text_input += " " + answer + self.tokenizer.eos_token

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

class VQABaseTrainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_file,
        transform=None,
        total_length=None,
        add_eos=None,
    ):
        super().__init__()
        self.transform = transform
        self.data_root = data_root
        self.annt_file = annt_file
        self.phase = 'train'
  
        if total_length is not None:
            self.annts = self.annts[:total_length]
        self.add_eos = add_eos
        self.ann = self.load_annotations()
        self.shuffle()
        print(f"length of the {self.__class__.__name__} is {len(self.ann)}")

    def load_annotations(self):
        raise NotImplementedError
    
    def shuffle(self):
        random.shuffle(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        try:
            image = self.loader(os.path.join(self.data_root, ann['file_name'])).convert('RGB')
            image = self.transform(image) if self.transform is not None else image
        except:
            print(os.path.join(self.data_root, ann['file_name']))
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        question = ann['question']
        answer = ann['answer']


        return image, question, answer
  
    def __len__(self):
        return len(self.ann)

class VQAV2TrainDataset(VQABaseTrainDataset):
    def __init__(
        self,
        data_root='./assets/coco/images',
        annt_root='./assets/VQAv2',
        ann_name_format='v2_mscoco_{split}2014_annotations.json',
        question_name_format='v2_OpenEnded_mscoco_{split}2014_questions.json',
        **kwargs,
    ):
        self.question_file = os.path.join(annt_root, question_name_format.format(split='train'))

        data_root = os.path.join(data_root, 'train2014')
        annt_file = os.path.join(annt_root, ann_name_format.format(split='train'))
        super().__init__(data_root=data_root, annt_file=annt_file, **kwargs)

    def load_annotations(self):
        answers_info = json.load(open(self.annt_file))['annotations']
        questions_info = json.load(open(self.question_file))['questions']

        annotations = {}
        for info in answers_info:
            image_id = info['image_id']
            question_id = info['question_id']
            answer = info['multiple_choice_answer'] if 'multiple_choice_answer' in info else info['answers'][0]['answer']

            assert question_id not in annotations
            annotations[question_id] = {
                'image_id': image_id,
                'question_id': question_id,
                'answer': answer,
                'file_name': f'COCO_{self.phase}2014_{image_id:012d}.jpg',
            }

        for info in questions_info:
            image_id = info['image_id']
            question_id = info['question_id']
            question = info['question']

            assert annotations[question_id]['image_id'] == image_id
            annotations[question_id]['question'] = question

        return list(annotations.values())

class OKVQATrainDataset(VQAV2TrainDataset):
    def __init__(
        self,
        annt_root='./assets/OK-VQA',
        ann_name_format='mscoco_{split}2014_annotations.json',
        question_name_format='OpenEnded_mscoco_{split}2014_questions.json',
        **kwargs,
    ):
        super().__init__(annt_root=annt_root, ann_name_format=ann_name_format, question_name_format=question_name_format, **kwargs)

class TextVQATrainDataset(VQABaseTrainDataset):
    def __init__(
        self,
        data_root='./assets/TextVQA/train_images',
        annt_root='./assets/TextVQA',
        annt_name_format='TextVQA_0.5.1_{split}.json',
        **kwargs,
    ):

        annt_file = os.path.join(annt_root, annt_name_format.format(split='train'))
        super().__init__(data_root=data_root, annt_file=annt_file, **kwargs)

    def load_annotations(self):
        meta_infos = json.load(open(self.annt_file))['data']

        annotations = {}
        for info in meta_infos:
            image_id = info['image_id']
            question_id = info['question_id']
            answer = random.choice(info['answers'])

            assert question_id not in annotations
            annotations[question_id] = {
                'image_id': image_id,
                'question_id': question_id,
                'answer': answer,
            }
            annotations[question_id]['question'] = info['question']
            annotations[question_id]['file_name'] = '{}.jpg'.format(info['image_id'])

        return list(annotations.values())

class GQATrainDataset(VQABaseTrainDataset):
    def __init__(
        self,
        data_root='./assets/gqa/images',
        annt_root='./assets/gqa',
        phase='train',
        ann_name_format='{split}_balanced_questions.json',
        **kwargs,
    ):
        annt_file = os.path.join(annt_root, ann_name_format.format(split=phase))
        super().__init__(data_root=data_root, annt_file=annt_file, **kwargs)

    def load_annotations(self):
        meta_info = json.load(open(self.annt_file))

        annotations = {}
        for idx,info in enumerate(meta_info):
            image_id = int(info['image'].split('.')[0])
            question_id = info['question_id']
            answer = info['answer']
            question=info['question']

            annotations[idx] = {
                'image_id': image_id,
                'question_id': question_id,
                'answer': answer,
                'file_name': info['image'],
                'question': question,
            }

        return list(annotations.values())
    
class AOKVQATrainDataset(VQABaseTrainDataset):
    def __init__(
        self,
        data_root='./assets/coco/images',
        annt_root='./assets/aokvqa',
        phase='train',
        ann_name_format='aokvqa_v1p0_{split}.json',
        **kwargs,
    ):
        annt_file = os.path.join(annt_root, ann_name_format.format(split=phase))
        super().__init__(data_root=data_root, annt_file=annt_file, **kwargs)

    def load_annotations(self):
        meta_info = json.load(open(self.annt_file))

        annotations = {}
        for info in meta_info:

            question_id = info['question_id']

            answer_key = "direct_answers"

            answer_weight = {}
            for answer in info[answer_key]:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(info[answer_key])
                else:
                    answer_weight[answer] = 1 / len(info[answer_key])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights

            assert question_id not in annotations
            annotations[question_id] = {
                'image_id': info['image_id'],
                'question_id': question_id,
                'answer': answer,
                'file_name': info["image"],
                'question': info["question"],
            }

        return list(annotations.values())
    
class OCRVQATrainDataset(VQABaseTrainDataset):
    def __init__(
        self,
        data_root='datasets/ocr_vqa/images',
        annt_root='datasets/ocr_vqa',
        phase='train',
        ann_name_format='dataset.json',
        **kwargs,
    ):
        annt_file = os.path.join(annt_root, ann_name_format)
        super().__init__(data_root=data_root, annt_file=annt_file, **kwargs)

    def load_annotations(self):
        meta_info = json.load(open(self.annt_file))

        processed_data = []

        for k in meta_info.keys():
            if meta_info[k]['split'] != 1: continue  # 1 for training, 2 for validation, 3 for test
            ext = os.path.splitext(meta_info[k]['imageURL'])[1]
            imageFile = k + ext
            assert len(meta_info[k]['questions']) == len(meta_info[k]['answers'])
            for q, a in zip(meta_info[k]['questions'], meta_info[k]['answers']):
                processed_data.append(
                    {'question': q,
                     'answer': a,
                     'file_name': imageFile,
                     'image_id': k,
                     }
                )

        return processed_data

class LLaVATrainDataset(BaseDataset):
    def __init__(
            self,
            annt_root,
            data_root,
            transform=None,
    ):
        super().__init__()
        self.annt_root = annt_root
        self.data_root = data_root
        self.transform = transform
        
        self.ann = []
        print("Formatting inputs...Skip in lazy mode")

        with open(os.path.join(self.annt_root,'new_llava_v1_5_mix665k.json'), 'r') as file:
            data = json.load(file)
            for item in data:
                try:
                    item['image'] = os.path.join(self.data_root,item['image'])
                    self.ann.append(item)
                except:
                    pass

            
        # split multi-round dialogues to single-round dialogue
        max_conv_num = 2  # 1 round
        print(f"data length before split: {len(self.ann)}")
        new_ann = []
        for item in self.ann:
            conversations = item["conversations"]
            conversations = [conversations[i:i + max_conv_num] for i in range(0, len(conversations), max_conv_num)]
            for conv in conversations:
                new_item = item.copy()
                if "<image>" not in conv[0]['value']:
                    conv[0]['value'] = "<image>\n" + conv[0]['value']
                new_item["conversations"] = conv
                new_ann.append(new_item)
        self.ann = new_ann
        print(f"data length after split: {len(self.ann)}")
    
    def __getitem__(self, index):
        while True:
            try:
                data = self.ann[index]
                
                assert len(data['conversations']) == 2
                
                query = data['conversations'][0]['value'].replace('<image>\n', '')
                query = query.replace('\n<image>', '')
                query = query.replace('<image>', '')
                
                image_id = data['id']
                image = self.loader(data['image']).convert('RGB')
                label = data['conversations'][1]['value']
                break
            except Exception as e:
                print(e)
                print('Error loading data:', data['image'])
                index = random.randint(0, len(self.ann) - 1)
        
        return self.transform(image), query, label
    
    def __len__(self):
        return len(self.ann)

class VQACocoCaptionKarpathyTrainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
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
        question=random.choice(QUESTIONS)
        image_name = item["image"]
        image_path = os.path.join(self.data_root, f"{image_name}")

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image,question,caption