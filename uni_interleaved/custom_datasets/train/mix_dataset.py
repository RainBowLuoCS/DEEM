"""Classes for mixing samples from multiple sources."""

import math
from itertools import permutations
import numpy as np
from typing import List
import torch
from torch.utils.data import IterableDataset, ConcatDataset

from ..utils.wds_utils import WdsDataset, pytorch_worker_info


def random_samples(datasets, probs=None, sampling_type="sum", seed=0, fix_seed=False, dataset_names=None):
    sources = [iter(d) for d in datasets]
    if probs is None:
        probs = [1] * len(sources)
    else:
        probs = list(probs)

    generator = torch.Generator()
    if not fix_seed:
        rank, world_size, worker, num_workers = pytorch_worker_info()
        seed += rank * num_workers + worker
    generator.manual_seed(seed)

    is_source_finished = [0] * len(sources)
    while len(sources) > 0 and sum(is_source_finished) < len(datasets):
        cum = (np.array(probs) / np.sum(probs)).cumsum()
        r = torch.rand(1, generator=generator).item()
        i = np.searchsorted(cum, r)
        try:
            data = next(sources[i])

            if dataset_names is not None:
                assert "meta" in data and isinstance(data["meta"], dict) and len(dataset_names) == len(datasets)
                data["meta"]["dataset_name"] = dataset_names[i]

            yield data
        except StopIteration:
            if sampling_type == "sum":
                del sources[i]
                del probs[i]
            elif sampling_type == "longest":
                sources[i] = iter(datasets[i])
                is_source_finished[i] = 1
            else:
                break


class RandomMixWdsDataset(IterableDataset):
    def __init__(
        self,
        datasets: List[WdsDataset],
        probs=None,
        sampling_type="sum",
        seed=0,
        fix_sampling_ratio=False,
        dataset_names=None,
    ):
        self.dataset_names = dataset_names
        self.datasets = datasets
        for dataset in datasets:
            try:
                dataset_len = len(dataset)
            except:
                dataset_len = -1
            
            dataset_name = getattr(dataset, 'dataset_name', dataset.__class__.__name__)
            print(f'{dataset_name}: {dataset_len}')
        
        self.fix_sampling_ratio = fix_sampling_ratio
        if self.fix_sampling_ratio:
            assert (
                probs is None
            ), "do not support setting different probs for each dataset when fixing sampling ratio."
            self._permute_dataset_by_rank()

        if probs is None:
            probs = [1] * len(datasets)
        else:
            probs = list(probs)

        self.probs = probs
        assert sampling_type in ["longest", "shortest", "sum"]
        self.sampling_type = sampling_type
        self.seed = seed

    def _permute_dataset_by_rank(self):
        permute_list = list(permutations(range(len(self.datasets))))
        rank, world_size, worker, num_workers = pytorch_worker_info()
        idx_list = permute_list[rank % len(permute_list)]
        self.datasets = [self.datasets[i] for i in idx_list]

    def __iter__(self):
        """Return an iterator over the sources."""
        return random_samples(
            self.datasets,
            self.probs,
            self.sampling_type,
            self.seed,
            fix_seed=self.fix_sampling_ratio,
            dataset_names=self.dataset_names,
        )

    def set_epoch(self, epoch):
        for d in self.datasets:
            d.set_epoch(epoch)

    def set_tokenizer(self, tokenizer):
        for d in self.datasets:
            d.set_tokenizer(tokenizer)

    @property
    def epoch(self):
        return self.datasets[0].epoch

    @property
    def tokenizer(self):
        return self.datasets[0].tokenizer

    def __repr__(self) -> str:
        repr_str = f"RandomMixDataset: probs={self.probs}; sampling_type={self.sampling_type}\n"
        for d in self.datasets:
            repr_str += repr(d) + "\n"
        return repr_str

    def __len__(self):
        try:
            lens_dataset = np.array([len(d) for d in self.datasets])
        except:
            # raise NotImplementedError
            return None

        if self.sampling_type == "sum":
            return sum(lens_dataset)
        elif self.sampling_type == "longest":
            i = np.argmax(lens_dataset)
            return math.ceil(lens_dataset[i] / self.probs[i] * sum(self.probs))
        else:
            i = np.argmin(lens_dataset)
            return math.ceil(lens_dataset[i] / self.probs[i] * sum(self.probs))

class DatasetWrapper(IterableDataset):
    def __init__(
        self,
        dataset,
        concat_mode: bool = False,
        max_len: int = 2048,
        max_img: int = 40,
        per_device_batch_size: int = 1,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = getattr(dataset, 'dataset_name', dataset.__class__.__name__)
        # self.collator = dataset.collator
        self.collator = None
        self.concat_mode = concat_mode
        self.max_len = max_len if concat_mode else 0
        self.per_device_batch_size = per_device_batch_size

        self.epoch = 0
        self.tokenizer = dataset.collator.tokenizer
        self.collator=EmptyCollator()

    @staticmethod
    def merge_cache(cache):
        merged_data = {}
        for key in cache[0]:
            merged_data[key] = cache[0][key]


        # NOTE we do not use this because of gt text ids  
        merged_data.pop('ignore_prompt_token_offset')
        merged_data.pop('meta')

        for data in cache[1:]:
            merged_data['image_tensors'] = torch.cat([merged_data['image_tensors'], data['image_tensors']], dim=0)
            merged_data['image_tensors_dec'] = torch.cat([merged_data['image_tensors_dec'], data['image_tensors_dec']], dim=0)
            merged_data['image_tensors_mask'] = torch.cat([merged_data['image_tensors_mask'], data['image_tensors_mask']], dim=0)
            merged_data['num_image_per_seq'] = merged_data['num_image_per_seq'] + data['num_image_per_seq']
            merged_data['text_ids'] = torch.cat([merged_data['text_ids'], data['text_ids']], dim=1)
            merged_data['attention_mask'] = torch.cat([merged_data['attention_mask'], data['attention_mask']], dim=1)
            merged_data['gt_text_ids'] = torch.cat([merged_data['gt_text_ids'], data['gt_text_ids']], dim=1)
        
        merged_data['concat_mode'] = True
        return merged_data

    def __iter__(self):
        assert self.dataset.collator is not None
        self.dataset.shuffle()
        
        cache = []
        yield_data = []
        cum_seq_len = 0
        cum_img_len = 0
        for data in self.dataset:
            inputs = self.dataset.collator([data])

            assert inputs['text_ids'].shape[0] == 1
            cum_seq_len += inputs['text_ids'].shape[1]
            if (cum_seq_len > self.max_len) and len(cache) > 0:
                yield_data.append(DatasetWrapper.merge_cache(cache))
                cache = [inputs]
                cum_seq_len = inputs['text_ids'].shape[1]
            else:
                cache.append(inputs)

            if len(yield_data) >= self.per_device_batch_size:
                yield self.dataset.collator(yield_data)
                yield_data = []

        if len(cache) > 0:
            yield_data.append(DatasetWrapper.merge_cache(cache))

        if len(yield_data) >= self.per_device_batch_size:
            yield self.dataset.collator(yield_data)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

class EmptyCollator:
    def __init__(self) -> None:
        pass
    
    def __call__(self, data_list):
        return data_list
        
class SameConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
    
class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.rand_tensor = torch.multinomial(self.weights, self.total_size, replacement=True).tolist()
    
    def __getitem__(self, idx):
        idx=self.rand_tensor[idx]
        return super().__getitem__(idx)
    
    def __len__(self):
        return self.total_size
    
    def shuffle(self,fix_seed=False):
        generator = torch.Generator()
        seed=0
        if not fix_seed:
            rank, world_size, worker, num_workers = pytorch_worker_info()
            seed += rank * num_workers + worker
        generator.manual_seed(seed)
        self.rand_tensor = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator).tolist()
 