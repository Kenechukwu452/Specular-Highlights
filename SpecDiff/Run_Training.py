#!/home/kogbuagu/pytorch_env/bin/python

import os
import sys
import Diffuser as diff
import Backbone
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.optim import Adam
import Trainer as T
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('./models', exist_ok=True)
#huge_loader, big_loader, small_loader = unifoil.get_and_load_dataset(batch_size=10, img_size=32)
image_size = 128
batchsize = 10
noise_steps = 200




epochs = 2000 

def save_path(model_name):
    return f'./models/128/{model_name}.pth'

#Models with presaved configs
UNetTran = Backbone.Flex(size = image_size,noise_steps = noise_steps).to(device)
UNetUViT = Backbone.UTFLEX(size = image_size,noise_steps = noise_steps).to(device)




def Diffusion_Train(save_path= save_path, predictor = None , loader = None, aifnetset = False): 
    diffuser = diff.CosSchDiffuser(steps=noise_steps, device="cuda")
    # Adjust timesteps and device as needed
    trainer = T.Trainer(model=predictor, diffuser=diffuser, data_loader = loader, epochs= epochs, lr=1e-4, device=device, aifnetset=aifnetset)
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

#Flex = Diffusion_Train(save_path, predictor = UNetTran, loader = aux_train, aifnetset=False)
# UTFLex = Diffusion_Train(save_path, predictor = UNetUViT, loader = aux_train, aifnetset=False)

#Flex = Diffusion_Train(save_path, predictor = UNetTran, loader = train_loader, aifnetset=True)
#UTFLex = Diffusion_Train(save_path, predictor = UNetUViT, loader = train_loader, aifnetset=True)