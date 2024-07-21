import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import CLIPModel


class RICES:
    def __init__(
        self,
        dataset,
        batch_size,
        vision_encoder_path="./assets/openai/clip-vit-large-patch14",
        cached_features_path=None,
        image_size=336,
    ):
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.image_size = image_size

        # Load the model and processor
        self.model = CLIPModel.from_pretrained(vision_encoder_path)

        cached_features_path = os.path.join(
            cached_features_path, f"{dataset.__class__.__name__}.pth"
        )

        # Precompute features
        if cached_features_path is None or not os.path.exists(cached_features_path):
            self.model = self.model.to(self.device)
            self.features = self._precompute_features()
            self.model = self.model.to("cpu")
            if dist.get_rank() == 0:
                os.makedirs(os.path.dirname(cached_features_path), exist_ok=True)
                torch.save(self.features, cached_features_path)
            dist.barrier()
        else:
            self.features = torch.load(cached_features_path, map_location="cpu")

    def _precompute_features(self):
        features = []

        # Switch to evaluation mode
        self.model.eval()

        def custom_collate_fn(data_list):
            images = []
            for data in data_list:
                image = data[0][0]
                images.append(
                    torch.from_numpy(image) if isinstance(image, np.ndarray) else image
                )
            return torch.stack(images)

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )
        from einops import rearrange    
        CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [
            0.26862954,
            0.26130258,
            0.27577711,
        ]
        mean, std = torch.tensor(CLIP_MEAN).to('cuda'), torch.tensor(CLIP_STD).to('cuda')
        mean, std = rearrange(mean, "c -> 1 c 1 1"), rearrange(std, "c -> 1 c 1 1")
        with torch.no_grad():
            for images in tqdm(
                loader,
                desc="Precomputing features for RICES",
            ):
                images = images.to(self.device)
                if images.shape[-1] != self.image_size:
                    images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
                image_features = self.model.get_image_features(pixel_values=(images-mean)/std)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                features.append(image_features.detach().cpu())

        features = torch.cat(features)
        return features

    def find(self, images, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            if images.ndim == 3:
                images = images.unsqueeze(0)

            if images.shape[-1] != self.image_size:
                images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            # Get the feature of the input image
            query_feature = self.model.get_image_features(pixel_values=images)
            query_feature = query_feature / query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]
