import os
import json
import argparse
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


# image_root="./datasets/mmc4/mmc4_images"
# new_image_root="./datasets/mmc4/mmc4_images"

def move_file(image_name,new_images_root):
    old_name=image_name
    new_name=os.path.join(new_images_root,image_name.split("/")[-1])
    cmd= f"mv {old_name} {new_name}"
    os.system(cmd)

def image_rename_script(args):
    os.makedirs(args.new_images_root,exist_ok=True)
    dir_names = [os.path.join(args.images_root,i) for i in os.listdir(args.images_root) if os.path.isdir(os.path.join(args.images_root,i))]
    file_names=[]
    for i in dir_names:
        file_names.extend([os.path.join(i,j) for j in os.listdir(i)])

    # functools.partial 用于固定 download_file 函数的 annt_root 参数
    partial_download_file = partial(move_file, new_images_root=args.new_images_root)

    # Adjust based on your machine's capability
    pool = Pool(processes=args.num_process)

    for _ in tqdm(pool.imap_unordered(partial_download_file, file_names), total=len(file_names)):
        pass

    pool.close()
    pool.join()

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--images_root", default="./datasets/mmc4/mmc4_images")
        parser.add_argument(
            "--new_images_root", default="./datasets/mmc4/mmc4_images_v2")

        parser.add_argument('--num_process', type=int, default=30,
                            help='Number of processes in the pool can be larger than cores')
        args = parser.parse_args()

        return args

    args = parse_args()
    image_rename_script(args)