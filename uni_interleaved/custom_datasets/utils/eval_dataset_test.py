import sys
sys.path.append("/home/luorun/workspace/NIPS")

from uni_interleaved.custom_datasets.eval.grounding_datasets import(
GroundingEvalCollator,
RefCOCOEvalDataset
)

from uni_interleaved.custom_datasets.eval.pairs_datasets import(
NoCapsEvalDataset,
Flickr30KEvalDataset,
Image2ParagraphEvalDataset,
CocoCaptionEvalDataset,
LNCOCOEvalDataset,
MSCOCOEvalDataset,
ImageTextPairEvalCollator
)

from uni_interleaved.custom_datasets.eval.vqa_datasets import(
VQAEvalCollator,
VQAV2EvalDataset,
OKVQAEvalDataset,
VizWizVQAEvalDataset,
TextVQAEvalDataset,
GQAEvalDataset
)

from uni_interleaved.custom_datasets.utils.build import create_transform
from torch.utils.data import DataLoader
import os

transform=create_transform("dual_numpy")

dataset=RefCOCOEvalDataset(data_root="datasets/coco/train2014", 
                            annt_root="datasets/refcoco",
                            split='refcoco_testA',
                            transform=transform)

dataset.dataset_name="refcoco_testA"
collator=GroundingEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('RefCOCO TestA Dataset Test Done!')

dataset=RefCOCOEvalDataset(data_root="datasets/coco/train2014", 
                            annt_root="datasets/refcoco",
                            split='refcoco_testB',
                            transform=transform)

dataset.dataset_name="refcoco_testB"
collator=GroundingEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('RefCOCO TestB Dataset Test Done!')

dataset=RefCOCOEvalDataset(data_root="datasets/coco/train2014", 
                            annt_root="datasets/refcoco",
                            split='refcoco_val',
                            transform=transform)

dataset.dataset_name="refcoco_val"
collator=GroundingEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('RefCOCO val Dataset Test Done!')

dataset=NoCapsEvalDataset(data_root="datasets/nocaps/val_imgs", 
                            annt_file="datasets/nocaps/nocaps_val_4500_captions.json",
                            transform=transform)

dataset.dataset_name="nocaps"
collator=ImageTextPairEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('Nocaps Dataset Test Done!')

dataset=Flickr30KEvalDataset(data_root="datasets/flickr30k/flickr30k-images", 
                            annt_file="datasets/flickr30k/flickr30k_test1k.json",
                            transform=transform)

dataset.dataset_name="flickr30k"
collator=ImageTextPairEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('Flickr30K Dataset Test Done!')

dataset=Image2ParagraphEvalDataset(data_root="datasets/vg", 
                            annt_root="datasets/image2parag",
                            transform=transform)

dataset.dataset_name="img2parag"
collator=ImageTextPairEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('Img2Parag Dataset Test Done!')

dataset=CocoCaptionEvalDataset(data_root="datasets/coco", 
                            annt_root="datasets/coco",
                            transform=transform)

dataset.dataset_name="cococaption"
collator=ImageTextPairEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('CocoCaption Dataset Test Done!')

dataset=LNCOCOEvalDataset(data_root="datasets/coco", 
                            annt_root="datasets/lncoco",
                            transform=transform)

dataset.dataset_name="lncoco"
collator=ImageTextPairEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('Lncoco Dataset Test Done!')

dataset=MSCOCOEvalDataset(data_root="datasets/coco", 
                            annt_root="datasets/coco",
                            transform=transform)

dataset.dataset_name="mscoco"
collator=ImageTextPairEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('Mscoco Dataset Test Done!')

dataset=VQAV2EvalDataset(data_root="datasets/coco", 
                            annt_root="datasets/vqav2",
                            transform=transform)

dataset.dataset_name="vqav2"
collator=VQAEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('VQAv2 Dataset Test Done!')

dataset=OKVQAEvalDataset(data_root="datasets/coco", 
                            annt_root="datasets/okvqa",
                            transform=transform)

dataset.dataset_name="okvqa"
collator=VQAEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('OKVQA Dataset Test Done!')


dataset=VizWizVQAEvalDataset(data_root="datasets/vizwiz", 
                            annt_root="datasets/vizwiz",
                            transform=transform)

dataset.dataset_name="vizwiz"
collator=VQAEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('VizWiz Dataset Test Done!')

dataset=TextVQAEvalDataset(data_root="datasets/textvqa/train_images", 
                            annt_root="datasets/textvqa",
                            transform=transform)

dataset.dataset_name="textvqa"
collator=VQAEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('TextVQA Dataset Test Done!')

dataset=GQAEvalDataset(data_root="datasets/gqa/images", 
                            annt_root="datasets/gqa",
                            transform=transform)

dataset.dataset_name="gqa"
collator=VQAEvalCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5")

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('GQA Dataset Test Done!')






