import os

# NOTE adjust the proxy address before run the download script 
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

from huggingface_hub import snapshot_download


allow_patterns=["*.model", "*.json", "*.bin",
"*.py", "*.md", "*.txt"]
ignore_patterns=["*.safetensors", "*.msgpack",
"*.h5", "*.ot",]

# LLM vicuna-7b/13b


# version = 'lmsys/vicuna-13b-v1.3'
version = 'lmsys/vicuna-7b-v1.5'

path = os.path.join('./assets', version)
os.makedirs(path, exist_ok=True)


snapshot_download(
    repo_id=version,
    repo_type='model',
    cache_dir="./assets",
    resume_download=True,
    max_workers=8,
    ignore_patterns=ignore_patterns
)

# VFM clip-vit/convnext-base/large
# version = "openai/clip-vit-large-patch14"
version = "openai/clip-vit-base-patch16"
path = os.path.join('./assets', version)
os.makedirs(path, exist_ok=True)

snapshot_download(
    repo_id=version,
    repo_type='model',
    cache_dir="./assets",
    resume_download=True,
    max_workers=8,
    ignore_patterns=ignore_patterns
)

# version= "laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup"
version= "laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K"
path = os.path.join('./assets', version)
os.makedirs(path, exist_ok=True)

snapshot_download(
    repo_id=version,
    repo_type='model',
    cache_dir="./assets",
    resume_download=True,
    max_workers=8,
    ignore_patterns=ignore_patterns
)


# DM stable diffusion v2.1
version = 'stabilityai/stable-diffusion-2-1-base'
path = os.path.join('./assets', version)
os.makedirs(path, exist_ok=True)

snapshot_download(
    repo_id=version,
    repo_type='model',
    cache_dir="./assets",
    resume_download=True,
    max_workers=8,
    ignore_patterns=ignore_patterns
)




