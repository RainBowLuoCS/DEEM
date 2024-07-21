import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from transformers.modeling_outputs import BaseModelOutputWithPooling
from open_clip.model import _build_vision_tower

DEFAULT_CONFIG={
        'convnext_base': {'timm_model_name': 'convnext_base', 'timm_model_pretrained': False, 'timm_pool': '', 'timm_proj': 'mlp', 'timm_drop': 0.0, 'timm_drop_path': 0.1, 'image_size': 360},
        'convnext_large': {'timm_model_name': 'convnext_large', 'timm_model_pretrained': False, 'timm_pool': '', 'timm_proj': 'mlp', 'timm_drop': 0.0, 'timm_drop_path': 0.1, 'image_size': 360}
}

class CLIPVisionConvNextAdapter(nn.Module):
    def __init__(self,config):
        super().__init__()
        if config['timm_model_name']=='convnext_base':
            embed_dim=640
            self.adapter_c2 = nn.Linear(128, 1024)
            self.adapter_c3 = nn.Linear(256, 1024)
            self.adapter_c4 = nn.Linear(512, 1024)
            self.adapter_c5 = nn.Linear(1024, 1024)
            self.adapter_cls = nn.Linear(1024,1024)
        elif config['timm_model_name']=='convnext_large':
            embed_dim=768
            self.adapter_c2 = nn.Linear(192, 1024)
            self.adapter_c3 = nn.Linear(384, 1024)
            self.adapter_c4 = nn.Linear(768, 1024)
            self.adapter_c5 = nn.Linear(1536, 1024)
            self.adapter_cls = nn.Linear(1536, 1024)
        else:
            raise NotImplementedError
        
        self.post_layernorm = nn.Identity()
        self.visual = _build_vision_tower(embed_dim, vision_cfg=config, quick_gelu=False)

    def forward(self, pixel_values=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict = None,):
        
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # c1 c2 c3 c4
        multiscale_features=[]
        x = self.visual.trunk.stem(pixel_values)
        feature_projs=[self.adapter_c2,self.adapter_c3,self.adapter_c4,self.adapter_c5]
        for i,feature_proj in zip(range(4),feature_projs):
            x = self.visual.trunk.stages[i](x)
            feature=feature_proj(x.permute(0,2,3,1)).permute(0,3,1,2)
            multiscale_features.append(feature)

        x = self.visual.trunk.norm_pre(x)
        cls=self.adapter_cls(self.visual.trunk.head(x))

        last_hidden_state = torch.cat([cls[:,None,], multiscale_features[-2].flatten(2).permute(0,2,1)], dim=1)

        pooled_output = cls
        pooled_output = self.post_layernorm(pooled_output)
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=cls,
            hidden_states=multiscale_features,
            attentions=None
        )


class CLIPVisionAdapterModel(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.vision_model = CLIPVisionConvNextAdapter(config)
        # Initialize weights and apply final processing
    def forward(
            self,
            pixel_values=None,
            output_attentions = None,
            output_hidden_states= None,
            return_dict= None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict
        
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

def clip_convnext_adapter_timm(
    model_path="assets/laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K",
    image_size=256,
    freeze=False,
    freeze_vit=False,
    gradient_checkpointing=True,
):
    if "convnext_base" in model_path:
        config=DEFAULT_CONFIG['convnext_base']
    elif "convnext_large" in model_path:
        config=DEFAULT_CONFIG['convnext_large']
    else:
        raise ValueError("path error!")
    
    model = CLIPVisionAdapterModel(config)
    message=model.vision_model.load_state_dict(torch.load(os.path.join(model_path,"open_clip_pytorch_model.bin")),
                                               strict=False)
    print(message)
    model.vision_model.visual.gradient_checkpointing = gradient_checkpointing
    # NOTE we do not use pooler output
    model.vision_model.post_layernorm.requires_grad_(False)
    print(f"Freeze clip_vit_adapter_hf is {freeze}")
    model.requires_grad_((not freeze))
    print(f"Freeze vit of clip_vit_adapter_hf is {freeze_vit}")
    if freeze_vit:
        for name, param in model.vision_model.named_parameters():
            if not name.startswith("adapter"):
                param.requires_grad_(False)
    
    return model
