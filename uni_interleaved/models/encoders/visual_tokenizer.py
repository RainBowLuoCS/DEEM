import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import random
import copy
import math
from tqdm import tqdm
from PIL import Image
import os
from .vit_adapter import clip_vit_adapter_hf
from .convnext_adapter import clip_convnext_adapter_timm
from ..decoders.perceiver import PerceiverResampler
from ..utils.pos_embed import get_abs_pos, get_2d_sincos_pos_embed
from ..decoders.sd import StableDiffusion

def tensor_to_pil(images: torch.Tensor):
    pil_images = images.mul(255).add_(0.5).clamp_(0, 255)
    pil_images = [
        Image.fromarray(img.permute(1, 2, 0).to("cpu", torch.uint8).numpy()).convert("RGB")
        for img in pil_images
    ]
    return pil_images

class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class MaskPooling(nn.Module):
    def __init__(self,pos_weight,neg_weight,mask_align):
        super().__init__()
        self.pos_weight=pos_weight
        self.neg_weight=neg_weight
        self.mask_align=mask_align
        self.mask_shape=112
        if self.mask_align:
            self.mask_linear=MLP(self.mask_shape*self.mask_shape,1024,1024,3)
            self.mask_feat_linear=nn.Linear(1024,1024)

    def extract(self, x, mask):
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

        pos_mask = (mask > 0).to(torch.bool)
        pos_denorm = pos_mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        neg_mask = ~pos_mask
        # neg_denorm = neg_mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = x*pos_mask*self.pos_weight

        return (mask_pooled_x/pos_denorm).sum(dim=(-1,-2))
    
    def extract_region(self, x, mask):
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

        pos_mask = (mask > 0).to(torch.bool)
        pos_denorm = pos_mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        neg_mask = ~pos_mask
        # neg_denorm = neg_mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = x*pos_mask*self.pos_weight

        return mask_pooled_x
    
    def forward(self, multi_scale_feats, mask):

        mask=mask.to(multi_scale_feats[0].dtype)
        shape_mask= F.interpolate(mask, size=self.mask_shape, mode='bilinear', align_corners=False)
        pos_feat = self.mask_linear(shape_mask.reshape(mask.shape[0], -1))
        mask_feats = mask.new_zeros(mask.shape[0],len(multi_scale_feats)+1, 1024)
        for idx,i in enumerate(multi_scale_feats):
            tmp_feat=self.extract(i,mask)
            tmp_feat=self.mask_feat_linear(tmp_feat.to(i.dtype))
            mask_feats[:,idx,:]=tmp_feat
        mask_feats[:,-1,:]=pos_feat

        return mask_feats
    

class VisualTokenizer(nn.Module):
    def __init__(
        self,
        sniffer_model_path="./assets/openai/clip-vit-large-patch14",
        perceiver_config=None,
        llm_hidden_size=5120,
        diffusion_hidden_size=1024,
        clip_normalize=True,
        grid_size=16,
        pretrained_model_name_or_path="",
        image_size=512,
        mmfs_input_channel=1024,
        mmfs_feat_levels=4,
        vae_encode_mini_bs=32,
        sd_base_seed=0,
        sd_use_random_seed=False,
        sd_use_vae_gradient_checkpointing=True,
        sd_use_unet_gradient_checkpointing=True,
        sd_use_encoder=False,
        freeze_vfm=False,
        freeze_dm=True,
        mask_align=False
    ) -> None:
        super().__init__()

        self.pos_weight=1
        self.neg_weight=0.0
        self.use_diffusion=sd_use_encoder
        self.use_diffusion_vit=True
        self.use_tta=False
        self.vis_inpaint=False
        self.mask_align=mask_align
        self.inpaint_num=0

        if self.use_diffusion:
            self.encoder= StableDiffusion(
                    pretrained_model_name_or_path,
                    image_size=image_size,
                    use_vae_gradient_checkpointing=sd_use_vae_gradient_checkpointing,
                    use_unet_gradient_checkpointing=sd_use_unet_gradient_checkpointing,
                    vae_encode_mini_bs=vae_encode_mini_bs,
                    base_seed=sd_base_seed,
                    use_random_seed=sd_use_random_seed,
                    mmfs_input_channel=mmfs_input_channel,
                    mmfs_feat_levels=mmfs_feat_levels,
                    freeze_dm=freeze_dm
                )

        self.clip_normalize = clip_normalize                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        self.grid_size=grid_size
        
        if "openai" in sniffer_model_path:
            self.sniffer = clip_vit_adapter_hf(model_path=sniffer_model_path,
                                               freeze_vit=freeze_vfm)
        elif 'laion' in sniffer_model_path:
            self.sniffer = clip_convnext_adapter_timm(model_path=sniffer_model_path,
                                                      freeze_vit=freeze_vfm)
        else:
            raise ValueError("No Surpport for Unkown Sniffer Format!")
        
        encoder_hidden_size = perceiver_config.encoder_hidden_size

        self.feature_extractor=MaskPooling(self.pos_weight,self.neg_weight,self.mask_align)

        self.pos_proj = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.pos_ln = nn.LayerNorm(encoder_hidden_size, eps=1e-6)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(
                get_2d_sincos_pos_embed(encoder_hidden_size, grid_size, cls_token=True)
            ).float()
        ).requires_grad_(False)

        self.perceiver_resampler = PerceiverResampler(**perceiver_config)
        self.length = perceiver_config.num_queries
        self.post_ln = nn.LayerNorm(encoder_hidden_size, eps=1e-6)

        self.proj_t = nn.Linear(perceiver_config.hidden_size, llm_hidden_size)
        self.proj_i = nn.Linear(perceiver_config.hidden_size, diffusion_hidden_size)

        self.initialize_weights()

        if self.clip_normalize:
            # normalize image
            CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [
                0.26862954,
                0.26130258,
                0.27577711,
            ]
            mean, std = torch.tensor(CLIP_MEAN), torch.tensor(CLIP_STD)
            mean, std = rearrange(mean, "c -> 1 c 1 1"), rearrange(std, "c -> 1 c 1 1")
            self.register_buffer("clip_mean", mean)
            self.register_buffer("clip_std", std)

    def print_parameters_stats(self, prefix=""):
        for name, module in self.named_children():
            print(
                f"# {prefix}{name} Total parameters: {sum(p.numel() for p in module.parameters()) / 1e6:.2f}M"
            )
            print(
                f"# {prefix}{name} Trainable parameters: {sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6:.2f}M"
            )

    def initialize_weights(self):
        nn.init.normal_(self.proj_t.weight, std=1.0e-3)
        nn.init.constant_(self.proj_t.bias, 0.0)
        nn.init.normal_(self.proj_i.weight, std=1.0e-3)
        nn.init.constant_(self.proj_i.bias, 0.0)

    def tta(self,image,image_dec,image_mask,samples_num=9,steps_num=10):
        with torch.enable_grad():
            # fix vis_embed result of the sniffer by feedback of the stable diffusion model 
            tmp_state_dict=copy.deepcopy(self.state_dict())
            scaler = torch.cuda.amp.GradScaler()
            optimizer = torch.optim.AdamW([{'params': i} for i in self.parameters()], lr=4e-6)
            interval_val = 1000 // samples_num

            for step in tqdm(range(steps_num)):
                start_point = random.randint(0, interval_val - 1)
                timesteps = torch.tensor(
                list(range(start_point, 1000, interval_val))*image.shape[0],
                device=image.device)
                model_output = self.sniffer(image)
                image_embed = model_output.last_hidden_state
                multiscale_features = model_output.hidden_states

                multiscale_features_n = []
                for ms_feat in multiscale_features:
                    pos_embed = get_abs_pos(
                        self.pos_embed[1:], ms_feat.size(2) * ms_feat.size(3)
                    )
                    pos_embed = rearrange(pos_embed, "(h w) c -> c h w", h=ms_feat.size(2))
                    ms_feat = ms_feat + pos_embed
                    multiscale_features_n.append(ms_feat)
                multiscale_features = multiscale_features_n

                pos_embed = get_abs_pos(self.pos_embed, image_embed.size(1))

                # inject mask knowledge
                B,_,D=image_embed.shape
                image_embed[:,1:]=self.feature_extractor(image_embed[:,1:].permute(0,2,1).reshape(B,D,self.grid_size,self.grid_size),
                                                        image_mask).reshape(B,D,-1).permute(0,2,1)

                qformer_inputs = self.pos_ln(self.pos_proj(image_embed))
                qformer_inputs = qformer_inputs + pos_embed

                # B,_,D=qformer_inputs.shape

                # qformer_inputs[:,1:]=self.feature_extractor(qformer_inputs[:,1:].permute(0,2,1).reshape(B,D,self.grid_size,self.grid_size),
                #                                           image_mask).reshape(B,D,-1).permute(0,2,1)
                
                image_embed = image_embed + pos_embed

                qformer_inputs = self.post_ln(qformer_inputs)
                vis_embed = self.perceiver_resampler(
                    encoder_hidden_states=qformer_inputs,
                    encoder_attention_mask=None,
                    return_dict=False,
                )[0]
                
                image_dec_h,image_dec_w=image_dec.shape[-2:]
                latent_image_mask=F.interpolate(image_mask, size=(image_dec_h//8,image_dec_w//8), mode='bilinear')
                latent_image_mask= (latent_image_mask > 0).to(torch.bool)

                mmfs_features=[self.feature_extractor(feat,image_mask) for feat in multiscale_features]
                
                # mmfs_features=[feat for feat in multiscale_features]
                mmfs_features=[feat[:, None] for feat in mmfs_features]
                mmfs_mask = torch.ones((vis_embed.shape[0], 1), dtype=torch.long, device=vis_embed.device)
                dif_embed = self.proj_i(vis_embed)
                sd_loss=self.encoder(torch.repeat_interleave(image_dec,samples_num,0), 
                                    torch.repeat_interleave(dif_embed,samples_num,0), 
                                    mmfs_features=[torch.repeat_interleave(i,samples_num,0) for i in mmfs_features],
                                    mmfs_mask=torch.repeat_interleave(mmfs_mask,samples_num,0),
                                    timesteps=timesteps, mode="encoder")
                
                full_mask=torch.repeat_interleave(torch.ones_like(latent_image_mask),samples_num,0)
                sniffer_loss=sd_loss*torch.repeat_interleave(latent_image_mask,samples_num,0)*self.pos_weight+self.neg_weight*sd_loss*torch.repeat_interleave(~latent_image_mask,samples_num,0)
                # sniffer_loss=sniffer_loss*torch.sum(full_mask,dim=(2,3), keepdim=True)/(torch.sum(latent_image_mask,dim=(2,3), keepdim=True)+1e-8)
                sniffer_loss=sniffer_loss.mean()
                print(f"step:={step},loss:={sniffer_loss.item()}")
                scaler.scale(sniffer_loss).backward()
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
            return tmp_state_dict
        
    def inpaint(self, image_dec, dif_embed, mmfs_features, mmfs_mask):
        image_b=tensor_to_pil(image_dec)
        for i, image in enumerate(image_b):
            image.save(
                os.path.join(
                    "vis/imagenet-a/origin", f"{i+self.inpaint_num}.png"
                )
            )

        negative_prompt_embeds = self.neg_prompt_embeds.expand_as(
            dif_embed
        )
        # print(dif_embed.shape,[i.shape for i in mmfs_features])
        # w=0.6
        # dif_embed=dif_embed*w+torch.flip(dif_embed, (0,))*(1-w)
        # # mmfs_mask*=0
        # mmfs_features=[i*(1-w)+torch.flip(i, (0,))*w for i in mmfs_features]
        image_ass=self.encoder.generate_inpaint_images(image_dec,dif_embed,
                                            negative_prompt_embeds=negative_prompt_embeds,
                                             mmfs_features=mmfs_features,
                                             mmfs_mask=mmfs_mask,
                                            num_inference_steps=250,
                                            guidance_scale=3.5,)
        image_a=tensor_to_pil(image_ass)
        print(image_ass.shape)
        for i, image in enumerate(image_a):
            image.save(
                os.path.join(
                    "vis/imagenet-a/ours", f"{i+self.inpaint_num}.png"
                )
            )
        self.inpaint_num+=len(image_a)

    def forward(self,image,image_dec,image_mask,samples_num=5,steps_num=5):

        if self.clip_normalize:
            # normalize image
            image = (image - self.clip_mean) / self.clip_std
        
        if self.use_tta and not self.training and self.use_diffusion:
            tmp_state_dict=self.tta(image,image_dec,image_mask,
                                    samples_num=samples_num,steps_num=steps_num)
            
        model_output = self.sniffer(image)
        image_embed = model_output.last_hidden_state
        # print(image_embed.shape)
        multiscale_features = model_output.hidden_states

        multiscale_features_n = []
        for ms_feat in multiscale_features:
            pos_embed = get_abs_pos(
                self.pos_embed[1:], ms_feat.size(2) * ms_feat.size(3)
            )
            pos_embed = rearrange(pos_embed, "(h w) c -> c h w", h=ms_feat.size(2))
            ms_feat = ms_feat + pos_embed
            multiscale_features_n.append(ms_feat)
        multiscale_features = multiscale_features_n

        pos_embed = get_abs_pos(self.pos_embed, image_embed.size(1))
        tmp_grid_size=int(math.sqrt(image_embed.size(1)))

        # inject mask knowledge
        B,_,D=image_embed.shape
        # image_embed[:,1:]=self.feature_extractor(image_embed[:,1:].permute(0,2,1).reshape(B,D,tmp_grid_size,tmp_grid_size),
        #                                           image_mask,multiscale_features).reshape(B,D,-1).permute(0,2,1)

        qformer_inputs = self.pos_ln(self.pos_proj(image_embed))
        qformer_inputs = qformer_inputs + pos_embed

        # B,_,D=qformer_inputs.shape

        qformer_inputs[:,1:]=self.feature_extractor.extract_region(qformer_inputs[:,1:].permute(0,2,1).reshape(B,D,tmp_grid_size,tmp_grid_size),
                                                  image_mask).reshape(B,D,-1).permute(0,2,1)
        image_embed = image_embed + pos_embed

        qformer_inputs = self.post_ln(qformer_inputs)
        if self.mask_align:
            mask_embed=self.feature_extractor(multiscale_features,image_mask)
            # qformer_inputs=torch.cat([qformer_inputs,mask_embed],dim=1)
            qformer_inputs=torch.cat([mask_embed,qformer_inputs],dim=1)
        vis_embed = self.perceiver_resampler(
            encoder_hidden_states=qformer_inputs,
            encoder_attention_mask=None,
            return_dict=False,
        )[0]
        
        image_dec_h,image_dec_w=image_dec.shape[-2:]
        latent_image_mask=F.interpolate(image_mask, size=(image_dec_h//8,image_dec_w//8), mode='bilinear')
        latent_image_mask= (latent_image_mask > 0).to(torch.bool)

        mmfs_features=[self.feature_extractor.extract_region(feat,image_mask) for feat in multiscale_features]
        
        # print([feat.shape for feat in mmfs_features])
        # mmfs_features=[feat[:,None] for feat in multiscale_features]
        mmfs_features=[feat[:, None] for feat in mmfs_features]
        mmfs_mask = torch.ones((vis_embed.shape[0], 1), dtype=torch.long, device=vis_embed.device)

        # mmfs_features = [
        #     torch.zeros_like(feat)[:, None] for feat in multiscale_features
        # ]
        # mmfs_mask = torch.zeros((vis_embed.shape[0], 1), dtype=torch.long, device=vis_embed.device)

        sniffer_loss=torch.tensor([0.0],device=vis_embed.device)
        
        if self.use_diffusion and self.use_diffusion_vit:
            if self.training:
                dif_embed = self.proj_i(vis_embed)

                sd_loss=self.encoder(image_dec, dif_embed, 
                                     mmfs_features=mmfs_features,
                                     mmfs_mask=mmfs_mask, mode="encoder")
                full_mask=torch.ones_like(latent_image_mask)
                sniffer_loss=sd_loss*latent_image_mask*self.pos_weight+self.neg_weight*sd_loss*(~latent_image_mask)
                sniffer_loss=sniffer_loss*torch.sum(full_mask,dim=(2,3), keepdim=True)/(torch.sum(latent_image_mask,dim=(2,3), keepdim=True)+1e-8)
                sniffer_loss=sniffer_loss.mean()
            if not self.training and self.vis_inpaint:
                dif_embed = self.proj_i(vis_embed)
                self.inpaint(image_dec,dif_embed, 
                            mmfs_features=mmfs_features,
                            mmfs_mask=mmfs_mask)

            # elif not self.training and self.use_tta:
            #     with torch.enable_grad():
            #         # fix vis_embed result of the sniffer by feedback of the stable diffusion model 
            #         vis_embed=nn.Parameter(vis_embed,requires_grad=True)
            #         tmp_state_dict=copy.deepcopy(self.encoder.state_dict())
            #         scaler = torch.cuda.amp.GradScaler()
            #         optimizer = torch.optim.AdamW([{'params': vis_embed}]+[{'params': i} for i in self.encoder.parameters()], lr=1e-4)
            #         interval_val = 1000 // samples_num
            #         start_point = random.randint(0, interval_val - 1)
            #         timesteps = torch.tensor(
            #             list(range(start_point, 1000, interval_val))*vis_embed.shape[0],
            #             device=vis_embed.device)
            #         for step in tqdm(range(steps_num)):
            #             dif_embed = self.proj_i(vis_embed)
            #             sd_loss=self.encoder(torch.repeat_interleave(image_dec,samples_num,0), 
            #                                 torch.repeat_interleave(dif_embed,samples_num,0), 
            #                                 mmfs_features=[torch.repeat_interleave(i,samples_num,0) for i in mmfs_features],
            #                                 mmfs_mask=torch.repeat_interleave(mmfs_mask,samples_num,0),
            #                                 timesteps=timesteps, mode="encoder")
            #             full_mask=torch.repeat_interleave(torch.ones_like(latent_image_mask),samples_num,0)
            #             sniffer_loss=sd_loss*torch.repeat_interleave(latent_image_mask,samples_num,0)*self.pos_weight+self.neg_weight*sd_loss*torch.repeat_interleave(~latent_image_mask,samples_num,0)
            #             # sniffer_loss=sniffer_loss*torch.sum(full_mask,dim=(2,3), keepdim=True)/(torch.sum(latent_image_mask,dim=(2,3), keepdim=True)+1e-8)
            #             sniffer_loss=sniffer_loss.mean()
            #             print(f"step:={step},loss:={sniffer_loss.item()}")
            #             scaler.scale(sniffer_loss).backward()
            #             scaler.step(optimizer)
            #             optimizer.zero_grad()
            #             scaler.update()
            #         self.encoder.load_state_dict(tmp_state_dict)

        vis_embed = self.proj_t(vis_embed)

        if self.use_tta and not self.training:
            self.load_state_dict(tmp_state_dict)

        output = dict(vis_embed=vis_embed)

        output["loss_sniffer"] = sniffer_loss
        output["image_embeds"] = image_embed[:, 1:, :]  # remove cls token
        output["multiscale_features"] = multiscale_features

        return output
    
