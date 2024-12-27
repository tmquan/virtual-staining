import warnings
# Suppress specific warnings related to image loading
# warnings.filterwarnings("ignore", category=UserWarning, message="Loading image.*with a slow-path.*")
warnings.simplefilter("ignore", category=Warning)

from contextlib import contextmanager, nullcontext

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from monai.losses import PerceptualLoss
from monai.utils import optional_import
from monai.metrics import PSNRMetric, SSIMMetric
tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

from monai.utils import optional_import
from monai.networks.nets import UNet, VNet, UNETR, SwinUNETR, BasicUNet, ViTAutoEnc, AttentionUnet
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

from typing import Any, Callable, Dict, Optional, OrderedDict, Tuple, List
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class LightningModule(LightningModule):
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig):
        super().__init__()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        

        self.unet2d_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            channels=[256, 256, 512],
            attention_levels=[False, False, True],
            num_head_channels=[0, 0, 512],
            num_res_blocks=2,
            with_conditioning=True, 
            cross_attention_dim=4, # Condition with dist, elev, azim, fov;  straight/hidden view  # flatR | flatT
            upcast_attention=True,
            use_flash_attention=True,
            use_combined_linear=True,
            dropout_cattn=0.5
        )
        init_weights(self.unet2d_model, init_type="normal", init_gain=0.01)

        # self.p20loss = None
        self.p20loss = PerceptualLoss(
            spatial_dims=2, 
            network_type="vgg", 
            is_fake_3d=False, 
            pretrained=True,
        ).eval() 
        
        if model_cfg.phase=="finetune":
            pass
        
        if self.train_cfg.ckpt:
            print("Loading.. ", self.train_cfg.ckpt)
            checkpoint = torch.load(self.train_cfg.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=False)
            # self.load_from_checkpoint(self.train_cfg.ckpt)

        self.save_hyperparameters()
        self.train_step_outputs = []
        self.validation_step_outputs = []

        self.psnr = PSNRMetric(max_val=1.0)
        self.ssim = SSIMMetric(spatial_dims=2, data_range=1.0)
        self.psnr_outputs = []
        self.ssim_outputs = []
    

    def forward_pix2pix_condition(self, image2d, context):
        _device = image2d.device
        B = image2d.shape[0]
        timesteps = 0*torch.randint(0, 1000, (B,), device=_device).long()  
        output = self.unet2d_model.forward(image2d, timesteps=timesteps, context=context)
        return output
    
    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        imageA = batch["imageA"]
        imageB = batch["imageB"]
        labelB = batch["labelB"].unsqueeze(-2) * 1.0
        _device = batch["imageA"].device
        B = imageA.shape[0]
        
        A_to_B = torch.cat([torch.zeros_like(labelB[...,[0]]), labelB[...,1:]], dim=-1) # Remove device prop, add direction
        B_to_A = torch.cat([torch.ones_like(labelB[...,[0]]), labelB[...,1:]], dim=-1) # Remove device prop, add direction

        estimAB = self.forward_pix2pix_condition(imageA, A_to_B)
        estimBA = self.forward_pix2pix_condition(imageB, B_to_A)
        
        estimABA = self.forward_pix2pix_condition(estimAB, B_to_A)
        estimBAB = self.forward_pix2pix_condition(estimBA, A_to_B)

        loss = self.train_cfg.alpha * F.l1_loss(estimAB, imageB) \
             + self.train_cfg.alpha * F.l1_loss(estimBA, imageA) \
             + self.train_cfg.alpha * F.l1_loss(estimBAB, imageB) \
             + self.train_cfg.alpha * F.l1_loss(estimABA, imageA) \
             
        if self.p20loss is not None:
            loss += self.train_cfg.lamda * self.p20loss(estimAB, imageB) \
                  + self.train_cfg.lamda * self.p20loss(estimBA, imageA) \
            
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B)
        
        # Visualization step
        if batch_idx == 0:
            # Sampling step for X-ray
            with torch.no_grad():
                viz2d = torch.cat([imageA, imageB, estimAB, estimABA, estimBA, estimBAB], dim=-1)
                tensorboard = self.logger.experiment
                grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(0, 1)
                tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * B + batch_idx)    
        return loss
                        
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="train")
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="validation")
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        imageA = batch["imageA"]
        imageB = batch["imageB"]
        labelB = batch["labelB"].unsqueeze(-2) * 1.0
        _device = batch["imageA"].device
        B = imageA.shape[0]

        A_to_B = torch.cat([torch.zeros_like(labelB), labelB[...,1:]], dim=-1)
        B_to_A = torch.cat([torch.ones_like(labelB), labelB[...,1:]], dim=-1)
        estimAB = self.forward_pix2pix_condition(imageA, A_to_B)
        estimABA = self.forward_pix2pix_condition(estimAB, B_to_A)

        estimBA = self.forward_pix2pix_condition(imageB, B_to_A)
        estimBAB = self.forward_pix2pix_condition(estimBA, A_to_B)
        # print(imageA.shape, imageB.shape, labelB.shape)
        psnr = self.psnr(estimAB, imageB)
        ssim = self.ssim(estimAB, imageB)
        self.psnr_outputs.append(psnr)
        self.ssim_outputs.append(ssim)
    
    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(
            f"train_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(
            f"validation_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        print(f"PSNR :{torch.stack(self.psnr_outputs).mean()}")
        print(f"SSIM :{torch.stack(self.ssim_outputs).mean()}")
        self.psnr_outputs.clear()
        self.ssim_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.unet2d_model.parameters(), lr=self.train_cfg.lr, betas=(0.5, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, #
            milestones=[100, 200, 300, 400], 
            gamma=0.5
        )
        return [optimizer], [scheduler]
