import os
import json
import random
import torch
import numpy as np

from ..utils.loader import BaseDataset
from ..utils.wds_utils import init_tokenizer
from ..utils.re_rank import RICES

class VQAEvalCollator:
    def __init__(
        self,
        tokenizer_path,
        num_img_token=32,
        generation_kwargs=None,
        instr_prompts=None,
        train_dataset=None,
        few_shot_n_shot=2,
        few_shot_template="Question: {question} Short answer: {answer}{eos_token}",
        use_rice=False,
        rice_encoder="./assets/openai/clip-vit-large-patch14",
        cached_features_path=None,
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
            "## USER: Based on the image, please answer the question. {image}{question} Please provide an accurate answer within one word.\n",
            "You are a helpful assistant.\n\n",
        ]

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
        return self._call_for_generate_texts(data_list)

    def _call_for_generate_texts(self, data_list):

        meta = []
        images_tensors_all = []
        images_tensors_dec_all = []
        num_image_per_seq = []
        text_inputs_with_prompt_image_all = []

        assis_prompt, user_prompt, sys_prompt = self.instr_prompts

        assert "{image}" in user_prompt and "{question}" in user_prompt

        use_few_shot = (
            "{few_shot_example}" in user_prompt and self.train_dataset is not None
        )

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor, question, answer, index = data

            assert isinstance(images_tensor, tuple)
            images_tensor, images_tensor_dec = images_tensor

            images_tensor = torch.from_numpy(images_tensor)
            images_tensor_dec = torch.from_numpy(images_tensor_dec)

            meta.append((index, question, answer))

            _images_tensor_all = [images_tensor]
            _image_tensors_dec_all = [images_tensor_dec]
            _num_image_per_seq = 1

            if use_few_shot:
                few_shot_example, images, images_dec = self.get_few_shot_samples(
                    query_image=images_tensor
                )
                text_input = user_prompt.format(
                    few_shot_example=few_shot_example,
                    image=self.image_subseq,
                    question=question,
                )
                # few-shot images first, then question image
                _images_tensor_all = images + _images_tensor_all
                _image_tensors_dec_all= images_dec + _image_tensors_dec_all

                _num_image_per_seq += len(images)
            else:
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

    def get_few_shot_samples(self, query_image=None):
        images = []
        images_dec = []

        if self.use_rice:
            samples = self.rice.find(query_image, self.few_shot_n_shot)[0]
        else:
            idx = random.sample(
                list(range(len(self.train_dataset))), self.few_shot_n_shot
            )
            samples = [self.train_dataset[i] for i in idx]

        few_shot_caption_only = "{image}" not in self.few_shot_template
        few_shot_image_only = "{question}" not in self.few_shot_template

        few_shot_example = ""
        for image, question, answer, _ in samples:
            if few_shot_caption_only:
                few_shot_example += self.few_shot_template.format(
                    question=question,
                    answer=answer,
                    eos_token="",
                )
            elif few_shot_image_only:
                images.append(
                    torch.from_numpy(image[0]) if isinstance(image[0], np.ndarray) else image[0]
                )

                images_dec.append(
                    torch.from_numpy(image[1]) if isinstance(image[1], np.ndarray) else image[1]
                )

                few_shot_example += self.few_shot_template.format(
                    image=self.image_subseq,
                )
            else:
                images.append(
                    torch.from_numpy(image[0]) if isinstance(image[0], np.ndarray) else image[0]
                )

                images_dec.append(
                    torch.from_numpy(image[1]) if isinstance(image[1], np.ndarray) else image[1]
                )

                few_shot_example += self.few_shot_template.format(
                    image=self.image_subseq,
                    question=question,
                    answer=answer,
                    eos_token="",
                )

        return few_shot_example, images, images_dec

class VQABaseEvalDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_file,
        transform=None,
        total_length=None,
        phase='val',
        collate_mode='generate_vqa',
    ):
        super().__init__()
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.annt_file = annt_file
        self.phase = phase
        self.ann = self.load_annotations()
        if total_length is not None:
            self.ann = self.ann[:total_length]
        print(f"length of the {self.__class__.__name__} is {len(self.ann)}")

    def load_annotations(self):
        raise NotImplementedError

    def __getitem__(self, index):
        ann = self.ann[index]
        image = self.loader(os.path.join(self.data_root, ann['file_name'])).convert('RGB')
        image = self.transform(image) if self.transform is not None else image
        question = ann['question']
        answer = ann['answer']
        question_id = ann.get('question_id', -1)

        return image, question, answer, question_id

    def __len__(self):
        return len(self.ann)

    @property
    def data_shape(self):
        return 4, 32, 32

class VQAV2EvalDataset(VQABaseEvalDataset):
    def __init__(
        self,
        data_root='./assets/coco/images',
        annt_root='./assets/VQAv2',
        phase='val',
        ann_name_format='v2_mscoco_{split}2014_annotations.json',
        question_name_format='v2_OpenEnded_mscoco_{split}2014_questions.json',
        **kwargs,
    ):
        self.question_file = os.path.join(annt_root, question_name_format.format(split=phase))

        data_root = os.path.join(data_root, f'{phase}2014')
        annt_file = os.path.join(annt_root, ann_name_format.format(split=phase))
        super().__init__(data_root=data_root, annt_file=annt_file, phase=phase, **kwargs)

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
        
        tmp=list(annotations.values())
        # random.shuffle(tmp)
        return tmp

class OKVQAEvalDataset(VQAV2EvalDataset):
    def __init__(
        self,
        annt_root='./assets/OK-VQA',
        ann_name_format='mscoco_{split}2014_annotations.json',
        question_name_format='OpenEnded_mscoco_{split}2014_questions.json',
        **kwargs,
    ):
        super().__init__(annt_root=annt_root, ann_name_format=ann_name_format, question_name_format=question_name_format, **kwargs)

class VizWizVQAEvalDataset(VQABaseEvalDataset):
    def __init__(
        self,
        data_root='./assets/VizWiz',
        annt_root='./assets/VizWiz-VQA',
        phase='val',
        **kwargs,
    ):
        data_root = os.path.join(data_root, phase)
        annt_file = os.path.join(annt_root, f'{phase}.json')
        super().__init__(data_root=data_root, annt_file=annt_file, phase=phase, **kwargs)

    def load_annotations(self):
        meta_info = json.load(open(self.annt_file))

        annotations = []
        for ann in meta_info:
            annotations.append({
                'question_id': int(ann['image'].split('_')[-1].split('.')[0]),
                'file_name': ann['image'],
                'question': ann['question'],
                'answer': ann['answers'][0]['answer'],
            })

        return annotations

class TextVQAEvalDataset(VQABaseEvalDataset):
    def __init__(
        self,
        data_root='./assets/TextVQA/train_images',
        annt_root='./assets/TextVQA',
        phase='val',
        ann_name_format='textvqa_{split}_annotations.json',
        question_name_format='textvqa_{split}_questions.json',
        **kwargs,
    ):
        self.question_file = os.path.join(annt_root, question_name_format.format(split=phase))
        annt_file = os.path.join(annt_root, ann_name_format.format(split=phase))
        super().__init__(data_root=data_root, annt_file=annt_file, phase=phase, **kwargs)

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
            }

        for info in questions_info:
            image = info['image']
            image_id = info['image_id']
            question_id = info['question_id']
            question = info['question']

            assert annotations[question_id]['image_id'] == image_id
            annotations[question_id]['question'] = question
            annotations[question_id]['file_name'] = image

        return list(annotations.values())

class GQAEvalDataset(VQABaseEvalDataset):
    def __init__(
        self,
        data_root='./assets/gqa/images',
        annt_root='./assets/gqa',
        phase='testdev',
        ann_name_format='{split}_balanced_questions.json',
        **kwargs,
    ):
        self.question_file =None
        annt_file = os.path.join(annt_root, ann_name_format.format(split=phase))
        super().__init__(data_root=data_root, annt_file=annt_file, phase=phase, **kwargs)

    def load_annotations(self):
        meta_info = json.load(open(self.annt_file))

        annotations = {}
        for info in meta_info:
            image_id = info['image'].split('.')[0]
            question_id = info['question_id']
            answer = info['answer']
            question=info['question']

            assert question_id not in annotations
            annotations[question_id] = {
                'image_id': image_id,
                'question_id': question_id,
                'answer': answer,
                'file_name': info['image'],
                'question': question,
            }

        return list(annotations.values())