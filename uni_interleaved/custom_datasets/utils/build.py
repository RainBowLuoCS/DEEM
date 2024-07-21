import torchvision.transforms as transforms
import numpy as np
import math
import random
from PIL import Image

# NOTE Train Dataset
from ..train.mmc4_wds import build_mmc4_webdataset
from ..train.laion_wds import build_laion_webdataset
from ..train.mix_dataset import RandomMixWdsDataset, WeightedConcatDataset, SameConcatDataset, DatasetWrapper
from ..train.vqa_datasets import (
    VQACaptionTrainCollator,
    VQACocoCaptionKarpathyTrainDataset,
    OKVQATrainDataset,
    VQAV2TrainDataset,
    TextVQATrainDataset,
    AOKVQATrainDataset,
    OCRVQATrainDataset,
    GQATrainDataset,
    LLaVATrainDataset
)
from ..train.pairs_datasets import(
    ImageTextPairTrainCollator,
    CocoCaptionKarpathyTrainDataset,
    Image2ParagraphTrainDataset,
    Flickr30kCaptionTrainDataset,
    MSCOCOTrainDataset,
    LNCOCOTrainDataset,
    TextCapsTrainDataset
)
from ..train.grounding_datasets import(
    GroundingTrainCollator,
    VisualGenomeTrainDataset,
    RefCOCOTrainDataset
)

from ..train.referring_mask_datasets import(
    ReferringMaskTrainCollator,
    ReferringVGDATA,
    ReferringVCRDataset,
    ReferringCOCODataset,
    ReferringRefCOCO,
    ReferringRefCOCOP,
    ReferringPascalPart,
    ReferringPartImagenet,
    OspreyConversations,
    OspreyLVISPosNeg,
    OspreyDetailedDescription,
    OspreyPartLevel,
    OspreyShortForm,
)
##########
# NOTE Evaluation Dataset

from ..eval.vqa_datasets import(
    VQAEvalCollator,
    VQAV2EvalDataset,
    TextVQAEvalDataset,
    VizWizVQAEvalDataset,
    GQAEvalDataset,
    OKVQAEvalDataset
)
from ..eval.pairs_datasets import(
    ImageTextPairEvalCollator,
    MSCOCOEvalDataset,
    NoCapsEvalDataset,
    Flickr30KEvalDataset,
    LNCOCOEvalDataset,
    CocoCaptionEvalDataset,
    Image2ParagraphEvalDataset,
)
from ..eval.grounding_datasets import(
    GroundingEvalCollator,
    RefCOCOEvalDataset
)

from..eval.referring_mask_datasets import(
    ReferringMaskEvalCollator,
    ReferringRefCOCOG
)

from ..eval.score_datasets import(
    ScoreEvalCollator,
    VisDialDenseEvalDataset,
)

from ..eval.imagenet_datasets import(
    ImageNetEvalDataset,
    POPEEvalDataset,
    ImageNetEvalCollator,
    VisEvalDataset
)


def create_transform(
    aug_type="numpy",
    resolution=224,
    resolution2=512,
    resize=True,
    center_crop=True,
    random_flip=True,
    scale=None,
):
    if aug_type == "numpy":
        assert resize
        transform = transform_numpy(
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip
        )

    elif aug_type.startswith("dual_"):
        aug_type = aug_type.replace("dual_", "")
        assert resolution2 > 0, f"{aug_type=}; {resolution2=}"
        transform = dual_transform(
            resolution1=resolution,
            resolution2=resolution2,
            aug_type=aug_type,
            resize=resize,
            scale=scale,
            center_crop=center_crop,
            random_flip=random_flip
        )

    return transform

class dual_transform:
    def __init__(
        self,
        resolution1,
        resolution2,
        aug_type="numpy",
        resize=False,
        center_crop=True,
        random_flip=True,
        scale=0.2,
    ):
        self.transform1 = create_transform(
            aug_type=aug_type,
            resolution=resolution1,
            resize=resize,
            random_flip=random_flip,
            center_crop=center_crop,
            scale=scale,
            resolution2=-1,
        )

        self.transform2 = create_transform(
            aug_type=aug_type,
            resolution=resolution2,
            resize=resize,
            random_flip=random_flip,
            center_crop=center_crop,
            scale=scale,
            resolution2=-1,
        )

    def __call__(self, pil_image):
        arr1 = self.transform1(pil_image)
        arr2 = self.transform2(pil_image)

        return arr1, arr2

    def __repr__(self):
        return f"Dual Transform: {self.transform1}\n{self.transform2}"

class transform_numpy:
    def __init__(
        self,
        resolution,
        center_crop=True,
        random_flip=True,
    ) -> None:
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip

    def __call__(self, pil_image):

        if self.center_crop:
            arr = center_crop_arr(pil_image, self.resolution)
        else:
            arr = np.array(
                pil_image.resize(
                    (self.resolution, self.resolution), resample=Image.BICUBIC
                )
            )

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32).transpose([2, 0, 1])

        # normalize to [0,1]
        arr = arr / 255.0

        return arr

    def __repr__(self):
        return (
            f"transform_numpy: {self.resolution=}, {self.center_crop=}, "
            f"{self.random_flip=}"
        )

def resize_arr(pil_image, image_size):
    pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    return arr

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def build_train_dataset(config):

    if isinstance(config, list):
        datasets = {}
        for _config in config:
            datasets[_config.name] = _build_train_dataset(_config)
        return datasets
    elif config.name == "random_mix":
        datasets = []
        for _config in config.datasets:
            datasets.append(_build_train_dataset(_config))
        dataset = RandomMixWdsDataset(
            datasets=datasets,
            probs=getattr(config, "probs", None),
            sampling_type=getattr(config, "sampling_type", "sum"),
            seed=getattr(config, "seed", 0),
            fix_sampling_ratio=getattr(config, "fix_sampling_ratio", False),
            dataset_names=getattr(config, "dataset_names", None),
        )
        dataset.collator = None
        return dataset
    elif config.name=="sft":
        # NOTE for caption\vqa\referring_mask sft process
        datasets = []
        for _config in config.datasets:
            datasets.append(_build_train_dataset(_config))

        dataset = SameConcatDataset(datasets=datasets)
        setattr(dataset, "tokenizer", datasets[0].collator.tokenizer)
        setattr(dataset,"collator",datasets[0].collator)
        return dataset
    
    elif config.name=="sft_grounding":
        # NOTE for grounding sft process
        datasets, lengths = [], []
        for _config in config.datasets:
            datasets.append(_build_train_dataset(_config))
            lengths.append(math.sqrt(len(datasets[-1])))
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        dataset = WeightedConcatDataset(datasets=datasets,weights=weights)
        setattr(dataset, "tokenizer", datasets[0].collator.tokenizer)
        setattr(dataset, "collator", datasets[0].collator)
        dataset= DatasetWrapper(dataset,
                                concat_mode=getattr(config, "concat_mode", True),
                                per_device_batch_size=getattr(config, "per_device_batch_size", 1))
        dataset.collator = None
        return dataset
    
    return _build_train_dataset(config)

def _build_train_dataset(config):
    transform = create_transform(**config.transform)

    if config.name == "coco_caption_karpathy":
        dataset = CocoCaptionKarpathyTrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
        )
    elif config.name == "image2paragraph":
        dataset = Image2ParagraphTrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
        )
    elif config.name == "flickr30kcaption":
        dataset = Flickr30kCaptionTrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
        )
    elif config.name == "mscoco":
        dataset = MSCOCOTrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
        )
    elif config.name == "lncoco":
        dataset = LNCOCOTrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
        )
    elif config.name == "textcaps":
        dataset = TextCapsTrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
        )
    elif config.name == "mmc4_wds":
        # Iterable dataset used for pretrain
        dataset = build_mmc4_webdataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            tokenizer_path=config.tokenizer_path,
            per_device_batch_size=config.per_device_batch_size,
            input_shards=config.input_shards,
            num_samples=config.num_samples,
            floor=getattr(config, "floor", False),
            seed=getattr(config, "seed", 42),
            num_workers=getattr(config, "num_workers", 1),
            num_img_token=config.num_img_token,
            max_num_images_per_seq=getattr(config, "max_num_images_per_seq", -1),
            loss_img_weight=getattr(config, "loss_img_weight", None),
            loss_txt_weight=getattr(config, "loss_txt_weight", None),
        )
    elif config.name == "laion_wds":
        # Iterable dataset used for pretrain
        dataset = build_laion_webdataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            tokenizer_path=config.tokenizer_path,
            per_device_batch_size=config.per_device_batch_size,
            input_shards=config.input_shards,
            num_samples=config.num_samples,
            floor=getattr(config, "floor", False),
            seed=getattr(config, "seed", 42),
            num_workers=getattr(config, "num_workers", 1),
            num_img_token=config.num_img_token,
            max_num_images_per_seq=getattr(config, "max_num_images_per_seq", -1),
            loss_img_weight=getattr(config, "loss_img_weight", None),
            loss_txt_weight=getattr(config, "loss_txt_weight", None),
        )
    elif config.name == "vqav2":
        dataset = VQAV2TrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
        )
    elif config.name == "okvqa":
        dataset = OKVQATrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
        )

    elif config.name == "textvqa":
        dataset = TextVQATrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
        )
    elif config.name == "llava":
        dataset = LLaVATrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
        )
    elif config.name == "aokvqa":
        dataset = AOKVQATrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
        )
    elif config.name == "ocrvqa":
        dataset = OCRVQATrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
        )
    elif config.name == "cocovqa":
        dataset = VQACocoCaptionKarpathyTrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
        )
    elif config.name == "gqa":
        dataset = GQATrainDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
        )
    elif config.name in (
        "refcoco",
        "refcoco+",
        "refcocog",
    ):
        dataset = RefCOCOTrainDataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            data_type=config.data_type,
            split_type=config.split_type,
            random_flip=getattr(config, "random_flip", False),
            random_resize_crop_prob=getattr(config, "random_resize_crop_prob", 0.0),
        )
    elif config.name == "vg":
        dataset = VisualGenomeTrainDataset(
            data_root=config.data_root,
            transform=transform,
            random_flip=getattr(config, "random_flip", False),
            random_resize_crop_prob=getattr(config, "random_resize_crop_prob", 0.0),
        )
    elif config.name == "mask_refcoco":
        dataset = ReferringRefCOCO(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "mask_refcocop":
        dataset = ReferringRefCOCOP(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "mask_coco":
        dataset = ReferringCOCODataset(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "mask_imagenet":
        dataset = ReferringPartImagenet(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "mask_pascal":
        dataset = ReferringPascalPart(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "ospreypartlevel":
        dataset = OspreyPartLevel(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "ospreylvisposneg":
        dataset = OspreyLVISPosNeg(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "ospreyconversations":
        dataset = OspreyConversations(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "ospreyshortform":
        dataset = OspreyShortForm(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "ospreydetaileddescription":
        dataset = OspreyDetailedDescription(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "mask_vg":
        dataset = ReferringVGDATA(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "mask_vcr":
        dataset = ReferringVCRDataset(
            ann_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    else:
        raise NotImplementedError(config.name)
    
    collator = build_data_collator(config)
    dataset.collator = collator
    if config.name not in ('mmc4_wds','laion_wds'):
        collator.train_dataset=dataset
    dataset.dataset_name = config.name

    if not hasattr(dataset, "tokenizer"):
        setattr(dataset, "tokenizer", dataset.collator.tokenizer)

    return dataset
    
def build_eval_dataset(config):
    if isinstance(config, list):
        datasets = {}
        for _config in config:
            datasets[_config.name] = _build_eval_dataset(_config)
        return datasets

    return _build_eval_dataset(config)

def _build_eval_dataset(config):
    transform = create_transform(**config.transform)

    if config.name == "coco_caption":
        dataset = CocoCaptionEvalDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
            collate_mode=getattr(config,"collate_mode", 'generate_texts')
        )
    elif config.name == "image2paragraph":
        dataset = Image2ParagraphEvalDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
            collate_mode=getattr(config,"collate_mode", 'generate_texts')
        )
    elif config.name == "flickr30kcaption":
        dataset = Flickr30KEvalDataset(
            data_root=config.data_root,
            annt_file=config.annt_file,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
            collate_mode=getattr(config,"collate_mode", 'generate_texts')
        )
    elif config.name == "mscoco":
        dataset = MSCOCOEvalDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
            collate_mode=getattr(config,"collate_mode", 'generate_images')
        )
    elif config.name == "lncoco":
        dataset = LNCOCOEvalDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
            collate_mode=getattr(config,"collate_mode", 'generate_images')
        )
    elif config.name == "nocaps":
        dataset = NoCapsEvalDataset(
            data_root=config.data_root,
            annt_file=config.annt_file,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", False),
            collate_mode=getattr(config,"collate_mode", 'generate_texts')
        )
    elif config.name == "vizwiz":
        dataset = VizWizVQAEvalDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            collate_mode=getattr(config,"collate_mode", 'generate_texts')
        )
    elif config.name == "vqav2":
        dataset = VQAV2EvalDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            collate_mode=getattr(config,"collate_mode", 'generate_vqa')
        )
    elif config.name == "okvqa":
        dataset = OKVQAEvalDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            collate_mode=getattr(config,"collate_mode", 'generate_vqa')
        )

    elif config.name == "textvqa":
        dataset = TextVQAEvalDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            collate_mode=getattr(config,"collate_mode", 'generate_vqa')
        )
    elif config.name == "gqa":
        dataset = GQAEvalDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            collate_mode=getattr(config,"collate_mode", 'generate_vqa')
        )
    elif config.name=="refcocog_mask_val":
        dataset = ReferringRefCOCOG(
            annt_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name in (
        "refcoco_val",
        "refcoco_testA",
        "refcoco_testB",
        "refcoco+_val",
        "refcoco+_testA",
        "refcoco+_testB",
        "refcocog_val",
        "refcocog_test",
    ):
        dataset = RefCOCOEvalDataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            split=config.name,
            collate_mode=getattr(config,"collate_mode", 'generate_grounding'),
            random_flip=getattr(config, "random_flip", False),
            random_resize_crop_prob=getattr(config, "random_resize_crop_prob", 0.0),
        )
    elif config.name=='visdial':
        dataset=VisDialDenseEvalDataset(
            tokenizer_path=config.tokenizer_path,
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            collate_mode=getattr(config,"collate_mode", 'generate_scores'),
            num_img_token=getattr(config, "num_img_token", 77),
        )
    elif config.name in ('imagenet-s','imagenet-r','imagenet-a','imagenetv2','imagenet-d'):
        dataset=ImageNetEvalDataset(
            tokenizer_path=config.tokenizer_path,
            annt_file=config.annt_file,
            data_root=config.data_root,
            phase=config.phase,
            transform=transform,
            collate_mode=getattr(config,"collate_mode", 'generate_imagenet'),
            num_img_token=getattr(config, "num_img_token", 77),
        )
    elif config.name in ('pope-a','pope-r','pope-p'):
        dataset=POPEEvalDataset(
            tokenizer_path=config.tokenizer_path,
            annt_file=config.annt_file,
            data_root=config.data_root,
            phase=config.phase,
            transform=transform,
            collate_mode=getattr(config,"collate_mode", 'generate_imagenet'),
            num_img_token=getattr(config, "num_img_token", 77),
        )
    elif config.name =='vis':
        dataset=VisEvalDataset(
            tokenizer_path=config.tokenizer_path,
            annt_file=config.annt_file,
            data_root=config.data_root,
            phase=config.phase,
            transform=transform,
            collate_mode=getattr(config,"collate_mode", 'generate_imagenet'),
            num_img_token=getattr(config, "num_img_token", 77),
        )
    else:
        raise NotImplementedError(config.name)
    
    collator = build_data_collator(config, dataset)
    dataset.collator = collator
    collator.train_dataset=dataset
    dataset.dataset_name = config.name

    
    if not hasattr(dataset, "tokenizer"):
        setattr(dataset, "tokenizer", dataset.collator.tokenizer)

    return dataset
    
def build_data_collator(config, train_dataset=None):
    collator_name = getattr(config, "collator", "")
    if not collator_name:
        return None
    if collator_name == "ImageTextPairTrainCollator":
        return ImageTextPairTrainCollator(
            tokenizer_path=config.tokenizer_path,
            uncond_prob=getattr(config, "uncond_prob", 0.0),
            num_img_token=getattr(config, "num_img_token", 32),
            img_first_prob=getattr(config, "img_first_prob", 1.0),
            padding=getattr(config, "padding", "longest"),
            train_dataset=train_dataset
        )
    elif collator_name == "ImageTextPairEvalCollator":
        return ImageTextPairEvalCollator(
            tokenizer_path=config.tokenizer_path,
            train_dataset=train_dataset,
            uncond_prob=getattr(config, "uncond_prob", 0.0),
            num_img_token=getattr(config, "num_img_token", 32),
            img_first_prob=getattr(config, "img_first_prob", 1.0),
            generation_kwargs=getattr(config, "generation_kwargs", None),
            instr_prompts=getattr(config, "instr_prompts", None),
            padding=getattr(config, "padding", "longest"),
            few_shot_n_shot=getattr(config, "few_show_n_shot", 2),
            few_shot_template=getattr(
                config,
                "few_shot_template",
                "Caption: {caption} {image}",
            ),
            use_rice=getattr(config, "use_rice", False),
            rice_encoder=getattr(
                config, "rice_encoder", "./assets/openai/clip-vit-large-patch14"
            ),
            cached_features_path=getattr(
                config, "cached_features_path", "./OUTPUT/cached_feature"
            ),
        )
    elif collator_name == "VQACaptionTrainCollator":
        return VQACaptionTrainCollator(
            tokenizer_path=config.tokenizer_path,
            num_img_token=getattr(config, "num_img_token", 32),
            train_dataset=train_dataset
        )
    elif collator_name == "VQAEvalCollator":
        return VQAEvalCollator(
            tokenizer_path=config.tokenizer_path,
            num_img_token=getattr(config, "num_img_token", 32),
            generation_kwargs=getattr(config, "generation_kwargs", None),
            instr_prompts=getattr(config, "instr_prompts", None),
            train_dataset=train_dataset,
            few_shot_n_shot=getattr(config, "few_shot_n_shot", 2),
            few_shot_template=getattr(
                config,
                "few_shot_template",
                "Question: {question} Short answer: {answer}{eos_token}",
            ),
            use_rice=getattr(config, "use_rice", False),
            rice_encoder=getattr(
                config, "rice_encoder", "./assets/openai/clip-vit-large-patch14"
            ),
            cached_features_path=getattr(
                config, "cached_features_path", "./OUTPUT/cached_feature"
            ),
        )
    elif collator_name == "GroundingTrainCollator":
        return GroundingTrainCollator(
            tokenizer_path=config.tokenizer_path,
            train_dataset=train_dataset,
            num_img_token=getattr(config, "num_img_token", 32),
            ignore_soi_token_loss=getattr(config, "ignore_soi_token_loss", None),
            ignore_bos2soi_token_loss=getattr(config, "ignore_bos2soi_token_loss", None),
            max_length=getattr(config, "max_length", 2048),
        )
    elif collator_name == "GroundingEvalCollator":
        return GroundingEvalCollator(
            tokenizer_path=config.tokenizer_path,
            task=getattr(config, "collate_task", "grounding"),
            num_img_token=getattr(config, "num_img_token", 32),
            generation_kwargs=getattr(config, "generation_kwargs", None),
            instr_prompts=getattr(config, "instr_prompts", None),
            ignore_soi_token_loss=getattr(config, "ignore_soi_token_loss", None),
            ignore_bos2soi_token_loss=getattr(config, "ignore_bos2soi_token_loss", None),
            max_length=getattr(config, "max_length", 2048),
        )
    elif collator_name == "ReferringMaskTrainCollator":
        return ReferringMaskTrainCollator(
            tokenizer_path=config.tokenizer_path,
            num_img_token=getattr(config, "num_img_token", 32),
            ignore_soi_token_loss=getattr(config, "ignore_soi_token_loss", None),
            ignore_bos2soi_token_loss=getattr(config, "ignore_bos2soi_token_loss", None),
            max_length=getattr(config, "max_length", 2048),)
    elif collator_name == "ReferringMaskEvalCollator":
        return ReferringMaskEvalCollator(
            tokenizer_path=config.tokenizer_path,
            num_img_token=getattr(config, "num_img_token", 32),
            ignore_soi_token_loss=getattr(config, "ignore_soi_token_loss", None),
            ignore_bos2soi_token_loss=getattr(config, "ignore_bos2soi_token_loss", None),
            max_length=getattr(config, "max_length", 2048),)
    
    elif collator_name == "ScoreEvalCollator":
        return ScoreEvalCollator()
    elif collator_name == "ImageNetEvalCollator":
        return ImageNetEvalCollator(
            tokenizer_path=config.tokenizer_path,
            num_img_token=getattr(config, "num_img_token", 32),
            generation_kwargs=getattr(config, "generation_kwargs", None),
            instr_prompts=getattr(config, "instr_prompts", None),
            train_dataset=train_dataset
        )
    return None