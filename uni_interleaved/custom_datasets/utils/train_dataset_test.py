import sys
sys.path.append("/home/luorun/workspace/NIPS")

from uni_interleaved.custom_datasets.train.grounding_datasets import(
RefCOCOTrainDataset,
VisualGenomeTrainDataset,
GroundingTrainCollator
)

from uni_interleaved.custom_datasets.train.pairs_datasets import(
MSCOCOTrainDataset,
LNCOCOTrainDataset,
CocoCaptionKarpathyTrainDataset,
TextCapsTrainDataset,
Flickr30kCaptionTrainDataset,
Image2ParagraphTrainDataset,
ImageTextPairTrainCollator
)

from uni_interleaved.custom_datasets.train.vqa_datasets import(
VQACocoCaptionKarpathyTrainDataset,
VQAV2TrainDataset,
OKVQATrainDataset,
AOKVQATrainDataset,
TextVQATrainDataset,
OCRVQATrainDataset,
GQATrainDataset,
VQACaptionTrainCollator
)

from uni_interleaved.custom_datasets.utils.build import create_transform
from torch.utils.data import DataLoader
import os

transform=create_transform("dual_numpy")

dataset=RefCOCOTrainDataset(data_root="datasets/coco/train2014", 
                            annt_root="datasets/refcoco", 
                            data_type="refcoco",
                            split_type="unc",
                            transform=transform)
dataset.dataset_name="refcoco"
collator=GroundingTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('RefCOCO Train Dataset Test Done!')

dataset=RefCOCOTrainDataset(data_root="datasets/coco/train2014", 
                            annt_root="datasets/refcoco", 
                            data_type="refcoco+",
                            split_type="unc",
                            transform=transform)
dataset.dataset_name="refcoco+"
collator=GroundingTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('RefCOCO+ Train Dataset Test Done!')

dataset=RefCOCOTrainDataset(data_root="datasets/coco/train2014", 
                            annt_root="datasets/refcoco", 
                            data_type="refcocog",
                            split_type="umd",
                            transform=transform)
dataset.dataset_name="refcocog"
collator=GroundingTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('RefCOCOg Train Dataset Test Done!')

dataset=VisualGenomeTrainDataset(data_root="datasets/vg",
                            transform=transform)
dataset.dataset_name="vg"
collator=GroundingTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('VG Train Dataset Test Done!')

dataset=MSCOCOTrainDataset(data_root="datasets/coco", 
                            annt_root="datasets/coco", 
                            transform=transform)

dataset.dataset_name="mscoco"
collator=ImageTextPairTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('MSCOCO Train Dataset Test Done!')


dataset=LNCOCOTrainDataset(data_root="datasets/coco", 
                            annt_root="datasets/lncoco", 
                            transform=transform)

dataset.dataset_name="lncoco"
collator=ImageTextPairTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('LNCOCO Train Dataset Test Done!')

dataset=CocoCaptionKarpathyTrainDataset(data_root="datasets/coco", 
                            annt_root="datasets/coco", 
                            transform=transform)

dataset.dataset_name="cocokarpathy"
collator=ImageTextPairTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('COCOKarpathy Train Dataset Test Done!')

dataset=TextCapsTrainDataset(data_root="datasets/textvqa/train_images", 
                            annt_root="datasets/textcaps", 
                            transform=transform)

dataset.dataset_name="textcaps"
collator=ImageTextPairTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('TextCaps Train Dataset Test Done!')

dataset=Flickr30kCaptionTrainDataset(data_root="datasets/flickr30k/flickr30k-images", 
                            annt_root="datasets/flickr30k", 
                            transform=transform)

dataset.dataset_name="flickr30k"
collator=ImageTextPairTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('Flickr30K Train Dataset Test Done!')

dataset=Image2ParagraphTrainDataset(data_root="datasets/vg", 
                            annt_root="datasets/image2parag", 
                            transform=transform)

dataset.dataset_name="ima2parag"
collator=ImageTextPairTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('Ima2Parag Train Dataset Test Done!')


dataset=VQACocoCaptionKarpathyTrainDataset(data_root="datasets/coco", 
                            annt_root="datasets/coco", 
                            transform=transform)

dataset.dataset_name="vqacoco"
collator=VQACaptionTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('vqacoco Train Dataset Test Done!')


dataset=VQAV2TrainDataset(data_root="datasets/coco", 
                            annt_root="datasets/vqav2", 
                            transform=transform)

dataset.dataset_name="vqav2"
collator=VQACaptionTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('vqav2 Train Dataset Test Done!')

dataset=OKVQATrainDataset(data_root="datasets/coco", 
                            annt_root="datasets/okvqa", 
                            transform=transform)

dataset.dataset_name="okvqa"
collator=VQACaptionTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('okvqa Train Dataset Test Done!')

dataset=AOKVQATrainDataset(data_root="datasets/coco", 
                            annt_root="datasets/aokvqa", 
                            transform=transform)

dataset.dataset_name="aokvqa"
collator=VQACaptionTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('aokvqa Train Dataset Test Done!')

dataset=TextVQATrainDataset(data_root="datasets/textvqa/train_images", 
                            annt_root="datasets/textvqa", 
                            transform=transform)

dataset.dataset_name="textvqa"
collator=VQACaptionTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('textvqa Train Dataset Test Done!')

dataset=OCRVQATrainDataset(data_root="datasets/ocrvqa/images", 
                            annt_root="datasets/ocrvqa", 
                            transform=transform)

dataset.dataset_name="ocrvqa"
collator=VQACaptionTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('ocrvqa Train Dataset Test Done!')

dataset=GQATrainDataset(data_root="datasets/gqa/images", 
                            annt_root="datasets/gqa", 
                            transform=transform)

dataset.dataset_name="gqa"
collator=VQACaptionTrainCollator(tokenizer_path="/home/luorun/workspace/LLaVA/lmsys/vicuna-7b-v1.5",
                                train_dataset=dataset)

loader=DataLoader(dataset,batch_size=100,shuffle=True,collate_fn=collator)
for i,x in enumerate(loader):
    print(i)
    if i>10:
        break

print('gqa Train Dataset Test Done!')



