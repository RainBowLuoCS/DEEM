
## please refer to https://github.com/Vision-CAIR/MiniGPT-4/blob/main/dataset/README_1_STAGE.md 
## for more details
## missing cc3m please refer to https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md
## https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip


import os
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from img2dataset import download
from tqdm.contrib.concurrent import process_map  # 使用tqdm的并行map功能
import json
import zipfile
import pandas as pd
import requests
import shelve
import magic
import glob
import subprocess
import time
import argparse
from functools import partial
from tqdm import tqdm
import math
# Set proxy if needed
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"


# #######################   images downloading   ##################### #

headers = {
    'User-Agent': 'Googlebot-Image/1.0',  # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}


IMAGE_EXTENSIONS = [
    "blp",
    "bmp",
    "dib",
    "bufr",
    "cur",
    "pcx",
    "dcx",
    "dds",
    "ps",
    "eps",
    "fit",
    "fits",
    "fli",
    "flc",
    "ftc",
    "ftu",
    "gbr",
    "gif",
    "grib",
    "h5",
    "hdf",
    "png",
    "apng",
    "jp2",
    "j2k",
    "jpc",
    "jpf",
    "jpx",
    "j2c",
    "icns",
    "ico",
    "im",
    "iim",
    "tif",
    "tiff",
    "jfif",
    "jpe",
    "jpg",
    "jpeg",
    "mpg",
    "mpeg",
    "msp",
    "pcd",
    "pxr",
    "pbm",
    "pgm",
    "ppm",
    "pnm",
    "psd",
    "bw",
    "rgb",
    "rgba",
    "sgi",
    "ras",
    "tga",
    "icb",
    "vda",
    "vst",
    "webp",
    "wmf",
    "emf",
    "xbm",
    "xpm",
]


def gather_image_info_shard(json_file):
    """Gather image info from shard"""
    data = []
    for sample_data in tqdm(json_file):
        # get image names from json
        sample_data = json.loads(sample_data)
        for img_item in sample_data['image_info']:
            data.append({
                'local_identifier': img_item['image_name'],
                'url': img_item['raw_url'],
            })
    return data


def save_status(args, shelve_filename):
    print(f'Generating Dataframe from results...')
    with shelve.open(shelve_filename) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)

    report_filename = os.path.join(
        args.report_dir, f'{args.shard_name}.tsv.gz')
    df.to_csv(report_filename, sep='\t',
              compression='gzip', header=False, index=False)
    print(f'Status report saved to {report_filename}')

    print('Cleaning up...')
    matched_files = glob.glob(f'{shelve_filename}*')
    for fn in matched_files:
        os.remove(fn)


def download_images_multiprocess(args, df, func):
    """Download images with multiprocessing"""

    chunk_size = args.chunk_size
    num_process = args.num_process

    print('Generating parts...')

    shelve_filename = '%s_%s_%s_results.tmp' % (
        args.shard_name, func.__name__, chunk_size)
    with shelve.open(shelve_filename) as results:

        pbar = tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = ((index, df[i:i + chunk_size], func) for index, i in enumerate(
            range(0, len(df), chunk_size)) if index not in finished_chunks)
        pbar.write(
            f'\t{int(len(df) / chunk_size)} parts. Using {num_process} processes.')

        pbar.desc = "Downloading"
        with Pool(num_process) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print(
        f'Finished downloading images for {args.input_jsonl}\nImages saved at {args.output_image_dir}')

    return shelve_filename


def call(cmd):
    subprocess.call(cmd, shell=True)


def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)


def _get_local_image_filename(row):
    return row['folder'] + '/' + row['local_identifier']


def download_image(row):
    fname = _get_local_image_filename(row)

    # Skip already downloaded images, retry others later
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
        return row

    try:
        # Use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(
            row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        rate_limit_idx = 0
        while response.status_code == 429:
            print(
                f'RATE LIMIT {rate_limit_idx} for {row["local_identifier"]}, will try again in 2s')
            response = requests.get(
                row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
            row['status'] = response.status_code
            rate_limit_idx += 1
            time.sleep(2)
            if rate_limit_idx == 5:
                print(
                    f'Reached rate limit for {row["local_identifier"]} ({row["url"]}). Will skip this image for now.')
                row['status'] = 429
                return row

    except Exception as e:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row

    if response.ok:
        try:
            with open(fname, 'wb') as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)

            # Resize image if it is too big
            call('mogrify -resize "800x800>" {}'.format(fname))

            # Use the following if mogrify doesn't exist or can't be found
            # img = Image.open(fname)
            # if max(img.size) > 800:
            #     img = img.resize((min(img.width, 800), min(img.height, 800)))
            #     img.save(fname)

            row['mimetype'] = magic.from_file(fname, mime=True)
            row['size'] = os.stat(fname).st_size
        except:
            # This is if it times out during a download or decode
            row['status'] = 408
            return row
        row['file'] = fname
    return row

def process_zip_file(args, idx):
    """Process a single ZIP file and download images."""
    args.input_jsonl = os.path.join(
        args.annt_root, "laion_shard_{}_v1.json".format(idx))
    
    if not os.path.exists(args.input_jsonl):
        return
    try:

        data = []

        # get image names from json
        sample_data = json.load(open(args.input_jsonl,'r'))
        for img_item in sample_data:
            data.append({
                'local_identifier': img_item['file_name'].split('/')[-1],
                'url': img_item['url'],
            })

        shard_folder = args.images_root + "/" + str(idx)
        # if not os.path.exists(shard_folder):
        os.makedirs(shard_folder,exist_ok=True)
        for d in data:
            d['folder'] = shard_folder
        df = pd.DataFrame(data)
        args.shard_name = idx
        shelve_filename = download_images_multiprocess(
            args=args,
            df=df,
            func=download_image,
        )
        save_status(args=args, shelve_filename=shelve_filename)
        
    except Exception as e:
        print(f"Error processing {args.input_jsonl}: {e}")

def image_download_scriptv2(args):

                
    # tmp_annt_file=f"{args.annt_root}/image_download_666666_tmp.json"

    # if not os.path.exists(tmp_annt_file):
    #     all_data=[]
    #     data=json.load(open(f"{args.annt_root}/ccs_synthetic_filtered_large.json","r"))
    #     all_data.extend(data)
    #     data=json.load(open(f"{args.annt_root}/laion_synthetic_filtered_large.json","r"))
    #     all_data.extend(data)      
    #     with open(tmp_annt_file, 'w') as f:
    #         json.dump(all_data,f)
    
    files_list=[f"{args.annt_root}/laion_shard_{shard_num}_v1.json" for shard_num in range(args.start_idx, args.end_idx)]
    download(
        processes_count=args.num_process,
        thread_count=args.threads,
        url_list=files_list,
        image_size=640,
        output_folder=args.images_root,
        resize_mode='keep_ratio_largest',
        resize_only_if_bigger=True,
        output_format="webdataset",
        input_format="json",
        url_col="url",
        enable_wandb=False,
        caption_col='caption',
        retries=1,
        skip_reencode=False,
        number_sample_per_shard=100000,
        distributor="multiprocessing",
        # user_agent_token=headers['User-Agent'],
        # incremental_mode='extend'
    )

def image_download_script(args):

    indices = [i for i in range(args.start_idx, args.end_idx)]

    with ProcessPoolExecutor(max_workers=args.num_process) as executor:
        futures = {executor.submit(process_zip_file, args, idx): idx for idx in indices}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
                print(f"Successfully processed shard {idx}.")
            except Exception as e:
                print(f"Error processing shard {idx}: {e}")

# ####################### annotations downloading ##################### #


# Function to delete zero-sized files in annt_root directory
def delete_zero_sized_files(annt_root):
    need_redownload = False
    for file in os.listdir(annt_root):
        file_path = os.path.join(annt_root, file)
        # Check if it's a file and has zero size
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            need_redownload = True
            os.remove(file_path)  # Delete the file
    return need_redownload


def download_file(annt_root):
    ## for ccs
    URL = f"https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered_large.json"
    ZIP_FILE = f"{annt_root}/ccs_synthetic_filtered_large.json"
    cmd = '''curl -fsSL --retry 3 --retry-delay 5 --max-time 20 --continue-at - "{}" -o "{}"'''.format(
        URL, ZIP_FILE)
    os.system(cmd)

    # for laion
    URL = f"https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_synthetic_filtered_large.json"
    ZIP_FILE = f"{annt_root}/laion_synthetic_filtered_large.json"
    cmd = '''curl -fsSL --retry 3 --retry-delay 5 --max-time 20 --continue-at - "{}" -o "{}"'''.format(
        URL, ZIP_FILE)
    os.system(cmd)

    ##### v1 

    # all_data=[]
    # data=json.load(open(f"{args.annt_root}/ccs_synthetic_filtered_large.json","r"))
    # all_data.extend(data)
    # data=json.load(open(f"{args.annt_root}/laion_synthetic_filtered_large.json","r"))
    # all_data.extend(data)  

    # import math

    # step=100000

    # for i in range(math.ceil(len(all_data)/step)):
    #     with open(f"{args.annt_root}/laion_shard_{i}_v1.json","w") as f:
    #         new_data=all_data[i*step:(i+1)*step]
    #         for sample_idx,sample in enumerate(new_data):
    #             ext=sample['url'].split('.')[-1].lower()
    #             if ext not in IMAGE_EXTENSIONS:
    #                 ext='jpg'
    #             sample['file_name']=f"{i}/{sample_idx:0>5d}.{ext}"
    #         json.dump(new_data,f)    


    ##### v2 avoid oom error

    import math
    import ijson

    step=100000
    shard_num=0
    new_data=[]
    captions=[]
    urls=[]

    data=ijson.parse(open(f"{args.annt_root}/ccs_synthetic_filtered_large.json","r"))

    for prefix,event,value in data:
        if prefix=='item.caption' and event=="string":
            captions.append(value)
        elif prefix=="item.url" and event=="string":
            urls.append(value)
        if len(captions)==step and len(urls)==step:
            for sample_idx,(sample_caption,sample_url) in enumerate(zip(captions,urls)):
                sample={'url':sample_url,'caption':sample_caption}
                ext=sample['url'].split('.')[-1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    ext='jpg'
                sample['file_name']=f"{shard_num}/{sample_idx:0>5d}.{ext}"
                new_data.append(sample)
            with open(f"{args.annt_root}/laion_shard_{shard_num}_v1.json","w") as f:
                json.dump(new_data,f)
            
            captions=[]
            urls=[]
            new_data=[]
            shard_num+=1

    data=ijson.parse(open(f"{args.annt_root}/laion_synthetic_filtered_large.json","r"))

    for prefix,event,value in data:
        if prefix=='item.caption' and event=="string":
            captions.append(value)
        elif prefix=="item.url" and event=="string":
            urls.append(value)
        if len(captions)==step and len(urls)==step:
            for sample_idx,(sample_caption,sample_url) in enumerate(zip(captions,urls)):
                sample={'url':sample_url,'caption':sample_caption}
                ext=sample['url'].split('.')[-1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    ext='jpg'
                sample['file_name']=f"{shard_num}/{sample_idx:0>5d}.{ext}"
                new_data.append(sample)
            with open(f"{args.annt_root}/laion_shard_{shard_num}_v1.json","w") as f:
                json.dump(new_data,f)
            
            captions=[]
            urls=[]
            new_data=[]
            shard_num+=1
    
    for sample_idx,(sample_caption,sample_url) in enumerate(zip(captions,urls)):
        sample={'url':sample_url,'caption':sample_caption}
        ext=sample['url'].split('.')[-1].lower()
        if ext not in IMAGE_EXTENSIONS:
            ext='jpg'
        sample['file_name']=f"{shard_num}/{sample_idx:0>5d}.{ext}"
        new_data.append(sample)
    with open(f"{args.annt_root}/laion_shard_{shard_num}_v1.json","w") as f:
        json.dump(new_data,f)


def annt_download_script(args):

    download_file(args.annt_root)

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--mode', required=True,
                            choices=["annt", "images"], help="download images or annotations")

        parser.add_argument(
            "--annt_root", default="./datasets/laion/laion_annts")
        parser.add_argument(
            "--images_root", default="./datasets/laion/laion_images")
        
        parser.add_argument(
            "--start_idx", type=int, default=0,
             help='start index of annt shard')
        
        parser.add_argument(
            "--end_idx", type=int, default=1775,
             help='end index of annt shard')
    
        parser.add_argument('--num_process', type=int, default=16,
                            help='Number of processes in the pool can be larger than cores')
        parser.add_argument('--threads', type=int, default=128,
                            help='Number of threads in the processes can be larger than cores')
        parser.add_argument('--chunk_size', type=int, default=5120,
                            help='Number of images per chunk per process')
        parser.add_argument('--shard_name', type=str, default=None)
        parser.add_argument('--report_dir', type=str, default='./status_report/',
                            help='Local path to the directory that stores the downloading status')

        args = parser.parse_args()

        return args

    args = parse_args()
    if args.mode == "annt":
        annt_download_script(args)
        while delete_zero_sized_files(args.annt_root):
            annt_download_script(args)
    else:
        image_download_scriptv2(args)