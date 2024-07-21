import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
from huggingface_hub import snapshot_download

allow_patterns=["*.model", "*.json", "*.bin",
"*.py", "*.md", "*.txt"]
ignore_patterns=["*.safetensors", "*.msgpack",
"*.h5", "*.ot",]

snapshot_download(
    repo_id="tonyqian/nocaps_val",
    repo_type='dataset',
    data_dir="./datasets/",
    resume_download=True,
    # proxies={"https": "http://127.0.0.1:7890"},
    max_workers=8,
    allow_patterns=["nocap_val_imgs.zip"],
    ignore_patterns=ignore_patterns
)

os.system("wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json .")

# import json
# import os

# a=json.load(open("./datasets/nocaps/nocaps_val_4500_captions.json",'r'))
# for i in a['images']:
#     if not os.path.exists(os.path.join("./datasets/nocaps/val_imgs/",i['file_name'])):
#         print(i)
# print(len(os.listdir("./datasets/nocaps/val_imgs/")))