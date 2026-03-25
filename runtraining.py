#!/usr/bin/env python3.12
import os
import sys
from sh_models import d_backbone as Backbone
from pipeline import Diffuser as diff
from pipeline import ResidualLoader as residual_loader



import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam
import specdiff_train as T
import random
from pathlib import Path




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
set_seed(42)

device = "cpu"# torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('./models', exist_ok=True)
#huge_loader, big_loader, small_loader = unifoil.get_and_load_dataset(batch_size=10, img_size=32)
image_size = 32
batchsize = 4
noise_steps = 200
threshold=0.9
soft_gamma=1.0
residual_mode="additive"
threshold_method="otsu"

    
"""diffuse_images = "/Users/27171653/Desktop/PhD/Highlight-modelling/PSD_Dataset/PSD_Dataset/PSD_Train/PSD_Train_diffuse"
glossy_images = "/Users/27171653/Desktop/PhD/Highlight-modelling/PSD_Dataset/PSD_Dataset/PSD_Train/PSD_Train_specular"
"""
diffuse_images = "/Users/27171653/Desktop/PhD/Highlight-modelling/PSD_Dataset/PSD_Dataset/PSD_Train/PSD_Train_diffuse"
glossy_images = "/Users/27171653/Desktop/PhD/Highlight-modelling/PSD_Dataset/PSD_Dataset/PSD_Train/PSD_Train_specular"



dataset, loader = residual_loader.make_residual_dataloader(
    diffuse_dir=diffuse_images,
    glossy_dir=glossy_images,
    batch_size=4,
    resize_hw=(image_size, image_size),
    soft_gamma=1.0,
    threshold_method="otsu",
    threshold=0.7,
    residual_mode=residual_mode,
)

epochs = 1000
model_path = "./checkpoints/model_checkpoint"
def save_path(model_name):
    return f'./models/128/{model_name}.pth'

#Models with presaved configs
UNetTran = Backbone.Flex(size = image_size,noise_steps = noise_steps).to(device)
UNetUViT = Backbone.UTFLEX(size = image_size,noise_steps = noise_steps).to(device)
UNet = Backbone.UNetWithAttention(depth=4).to(device)
lantentUNet = Backbone.LatentUNetWithAttention(depth=4).to(device)

#combined_dataset, aux_train, aux_test, means, stds = prep.get_and_load_dataset()


def Diffusion_Train(save_path= save_path, 
                    predictor = None , 
                    loader=None, 
                    latent="false",
                    device=device, 
                    vae=None, 
                    vae_scale=None,
                    loss_type="weighted"): 
    diffuser = diff.CosSchDiffuser(steps=noise_steps, device=device)
    # Adjust timesteps and device as needed
    trainer = T.Trainer(model=predictor, 
                        diffuser=diffuser, 
                        data_loader = loader, 
                        epochs= epochs, 
                        lr=1e-4, 
                        device=device, 
                        latent=latent,
                        type=loss_type)
    # Start training
    Trained_model = trainer.train()
    path = str(save_path(predictor.__class__.__name__))

    Trained_model = Trained_model["last_model"]
    Optimizer =  trainer.optimizer.state_dict()
    # Save the model
    torch.save({
                    'model_state_dict': Trained_model.state_dict(),
                    'optimizer_state_dict': Optimizer,
                    }, path)
    print(f"Model saved to {path}")

    return Trained_model



SpecDiff = Diffusion_Train(save_path, predictor = UNet, loader = loader)
LatentSpecDiff = Diffusion_Train(save_path, predictor = lantentUNet, loader = loader, latent="true")



