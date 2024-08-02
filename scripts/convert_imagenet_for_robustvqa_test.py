import os
import glob
import json
import csv
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, CLIPModel, AutoProcessor
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

all_data={}
imagenet_d=[]
# # imagenet_d
# csv_reader = csv.reader(open("datasets/robustvqa/imagenet-d/questions/background.csv"), delimiter='\t')

# for idx,row in enumerate(csv_reader):
#     if idx==0:
#         continue
#     imagenet_d.append({'file_name':'datasets/robustvqa/imagenet-d/'+row[0],'gt_caption':row[1],'error_caption':row[2]})

# csv_reader = csv.reader(open("datasets/robustvqa/imagenet-d/questions/material.csv"), delimiter='\t')
# for idx,row in enumerate(csv_reader):
#     if idx==0:
#         continue
#     imagenet_d.append({'file_name':'datasets/robustvqa/imagenet-d/'+row[0],'gt_caption':row[1],'error_caption':row[2]})

# csv_reader = csv.reader(open("datasets/robustvqa/imagenet-d/questions/texture.csv"), delimiter='\t')

# for idx,row in enumerate(csv_reader):
#     if idx==0:
#         continue
#     imagenet_d.append({'file_name':'datasets/robustvqa/imagenet-d/'+row[0],'gt_caption':row[1],'error_caption':row[2]})
     
# all_data['imagenet_d']=imagenet_d

class_index_mapping = json.load(open("datasets/robustvqa/imagenet_class_index.json",'r'))
new_class_index_mapping = {}
classes=[]
index = 0
for key,val in class_index_mapping.items():
    new_class_index_mapping[val[0]] =[key,val[1]]
    assert int(key) == index
    index +=1
    classes.append(val[1])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,path_list,class_list,type_list):
        self.path_list=path_list
        self.class_list=class_list
        self.type_list=type_list

    def __len__(self,):
        return len(self.type_list)
    
    def __getitem__(self,index):
        
        image=Image.open(self.path_list[index]).convert('RGB').resize(
                    (image_size,image_size), resample=Image.BICUBIC
                )
        image=np.array(image)
        image=image.astype(np.float32).transpose([2, 0, 1])/255.0

        return image, self.class_list[index], self.type_list[index], self.path_list[index]
from einops import rearrange    
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [
    0.26862954,
    0.26130258,
    0.27577711,
]
mean, std = torch.tensor(CLIP_MEAN).to('cuda'), torch.tensor(CLIP_STD).to('cuda')
mean, std = rearrange(mean, "c -> 1 c 1 1"), rearrange(std, "c -> 1 c 1 1")

model_path="/mnt/workspace/lr/datasets/checkpoints/openai/clip-vit-large-patch14-336/"
model = CLIPModel.from_pretrained(model_path)
model.eval()
model=model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

inputs = tokenizer([f"A photo of a {c}" for c in classes], padding=True, return_tensors="pt")
inputs={k:v.to('cuda') for k,v in inputs.items()}
text_features = model.get_text_features(**inputs)

images=[]
labels=[]
types=[]
# imagenet_a
images_a=sorted(glob.glob(os.path.join("datasets/robustvqa/imagenet-a", '*/*.jpg')))
labels_a=[int(new_class_index_mapping[i.split('/')[-2]][0]) for i in images_a]
types_a=['imagenet-a']*len(labels_a)

images+=images_a
labels+=labels_a
types+=types_a
# imagenet_r
images_r=sorted(glob.glob(os.path.join("datasets/robustvqa/imagenet-r", '*/*.jpg')))
labels_r=[int(new_class_index_mapping[i.split('/')[-2]][0]) for i in images_r]
types_r=['imagenet-r']*len(labels_r)

images+=images_r
labels+=labels_r
types+=types_r

# # imagenet_s
# images_s=sorted(glob.glob(os.path.join("datasets/robustvqa/imagenet-s", '*/*.png')))
# labels_s=[int(new_class_index_mapping[i.split('/')[-2]][0]) for i in images_s]
# types_s=['imagenet-s']*len(labels_s)

# images+=images_s
# labels+=labels_s
# types+=types_s

# imagenet_v2
images_v2=sorted(glob.glob(os.path.join("datasets/robustvqa/imagenetv2", '*/*.jpeg')))
labels_v2=[int(i.split('/')[-2]) for i in images_v2]
types_v2=['imagenetv2']*len(labels_v2)

images+=images_v2
labels+=labels_v2
types+=types_v2

dataset=MyDataset(images,labels,types)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=512,
)
image_size=336

imagenet_a=[]
imagenet_r=[]
imagenet_s=[]
imagenet_v2=[]

with torch.no_grad():
    for images,labels,types,paths in tqdm(
        loader,
        desc="Precomputing features for Rank",
    ):
        images = images.to('cuda')
        if images.shape[-1] != image_size:
            images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
        image_features = model.get_image_features(pixel_values=(images-mean)/std)
        # image_features = model.get_image_features(pixel_values=images)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        output = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, pred = output.topk(2, 1, True, True)
        pred=pred.detach().cpu()
        flag=(pred[:,0].view(-1)==labels.view(-1))
        res=pred[~flag,0]
        # res=pred[:,0]
        # print((pred[:,0].view(-1)==labels.view(-1)).sum()/32*100)
        for itype,label,ss,path in zip(types,labels[~flag],res,paths):
            item={"file_name":path,
                  'gt_caption':class_index_mapping[str(label.item())][1],
                  'error_caption':class_index_mapping[str(ss.item())][1]}
            # print(item)
            if itype=="imagenet-a":
                imagenet_a.append(item)
            elif itype=="imagenet-r":
                imagenet_r.append(item)
            elif itype=="imagenetv2":
                imagenet_v2.append(item)

all_data['imagenet_a']=imagenet_a
all_data['imagenet_r']=imagenet_r
all_data['imagenet_v2']=imagenet_v2

with open("datasets/robustvqa/robustvqa_test.json", 'w') as f:
    json.dump(all_data,f)
