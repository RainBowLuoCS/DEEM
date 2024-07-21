import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
from huggingface_hub import snapshot_download

allow_patterns=["*.model", "*.json", "*.bin",
"*.py", "*.md", "*.txt"]
ignore_patterns=["*.safetensors", "*.msgpack",
"*.h5", "*.ot",]

snapshot_download(
    repo_id="nlphuji/flickr30k",
    repo_type='dataset',
    cache_dir="/home/luorun/datasets",
    resume_download=True,
    # proxies={"https": "http://127.0.0.1:7890"},
    max_workers=8,
    allow_patterns=["flickr30k-images.zip"],
    ignore_patterns=ignore_patterns
)