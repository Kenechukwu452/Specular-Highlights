#!/usr/bin/env python3.12
import os
import sys
from sh_models import d_backbone as Backbone
from pipeline import Diffuser as diff
from pipeline import difresloader
from torch.utils.data import DataLoader, Subset



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
image_size = 64
batchsize = 4
noise_steps = 200
threshold=0.9
soft_gamma=1.0
residual_mode="additive"
threshold_method="otsu"

project_root = Path(__file__).resolve().parent
default_dataset_root = project_root.parent / "PSD_Dataset" / "PSD_Dataset" / "PSD_Train"
dataset_root = Path(os.environ.get("PSD_TRAIN_ROOT", default_dataset_root)).expanduser()

diffuse_images = dataset_root / "PSD_Train_diffuse"
glossy_images = dataset_root / "PSD_Train_specular"

if not diffuse_images.exists() or not glossy_images.exists():
    raise FileNotFoundError(
        "Could not find PSD dataset folders. "
        f"Expected {diffuse_images} and {glossy_images}. "
        "Set the PSD_TRAIN_ROOT environment variable to the PSD_Train directory if needed."
    )



dataset, loader = difresloader.make_diffusion_dataloader(
    diffuse_dir=diffuse_images,
    glossy_dir=glossy_images,
    batch_size=4,
    resize_hw=(image_size, image_size),
    residual_mode=residual_mode,
    augment=True,
    random_hflip_p=0.5,
    random_vflip_p=0.0,
    enable_rot90=True,
)

small_dataset = Subset(dataset, list(range(1, 20)))  # position_06.png for quick testing

small_loader = DataLoader(
    small_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

epochs = 1000
model_path = "./checkpoints/model_checkpoint"
def save_path(model_name):
    return f'./models/{model_name}.pth'

#Models with presaved configs
UNetTran = Backbone.Flex(size = image_size,noise_steps = noise_steps).to(device)
UNetUViT = Backbone.UTFLEX(size = image_size,noise_steps = noise_steps).to(device)
UNet = Backbone.UNetWithAttention(depth=4).to(device)

#combined_dataset, aux_train, aux_test, means, stds = prep.get_and_load_dataset()


def Diffusion_Train(save_path= save_path, 
                    predictor = None , 
                    loader=None, 
                    device=device, 
                    loss_type="e"): 
    diffuser = diff.CosSchDiffuser(steps=noise_steps, device=device)
    # Adjust timesteps and device as needed
    trainer = T.Trainer(model=predictor, 
                        diffuser=diffuser, 
                        data_loader = loader, 
                        epochs= epochs, 
                        lr=1e-4, 
                        device=device, 
                        objective=loss_type)
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



SpecDiff = Diffusion_Train(save_path, predictor = UNet, loader = small_loader)

