import json


origin=json.load(open("datasets/llava/llava_v1_5_mix665k.json","r"))
new=[]
for i in origin:
    if 'image' not in i:
        continue
    new.append(i)
json.dump(new,open("datasets/llava/new_llava_v1_5_mix665k.json","w"))

    