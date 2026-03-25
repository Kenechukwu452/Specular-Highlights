import os
import datetime
from copy import deepcopy
from collections import OrderedDict
import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from diffusers.models import AutoencoderKL #Importing VAE from Hugging face diffusers library
# from your_model_file import UViT, Diffuser  # <-- Make sure to import your model and diffuser here

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(wait=False): pass  # fallback if not in notebook

residual_mode = "subtractive"  # change to "subtractive" for diffuse = glossy - residual
MASK_SOFT_FOCUS_GAMMA = 2.0
RESIDUAL_OUTPUT_SCALE = 0.5


def get_cosine_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    def cosine_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return idx_epoch / warmup_epoch
        else:
            return 1-(1-(math.cos((idx_epoch-warmup_epoch)/(epochs-warmup_epoch)*math.pi)+1)/2)*(1-final_lr/initial_lr)
    return cosine_lambda

def Error_log(model_name, error_message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("./logs", exist_ok=True)
    log_file = "./logs/error_log.txt"
    
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] Error in model: {model_name}\n")
        f.write(f"Message: {error_message}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Error logged to {log_file}")


def Checkpoint_save(model, loss, l_epoch, save_path, Parameterization = "e"):
    base_dir = os.path.dirname(save_path)
    os.makedirs(base_dir, exist_ok=True)
    model_name = model.__class__.__name__

    # Save loss to CSV
    loss_file = os.path.join(base_dir, f'Uncert{model_name}_loss.csv')
    with open(loss_file, 'a') as f:
        f.write(f"{l_epoch},{loss:.6f}\n")

    # Save model weights
    model_dir = os.path.join(base_dir, f"Uncert{model_name}_epoch_{l_epoch}_TParam_{Parameterization}_loss_{loss:.4f}")
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_file = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Model checkpoint saved to {checkpoint_file}")

    # Save architecture
    arch_file = os.path.join(model_dir, "architecture.txt")
    with open(arch_file, 'w') as f:
        f.write(str(model))

def weighted_mse_loss(prediction, target, mask):
    mse_loss = F.mse_loss(prediction, target, reduction='none')
    weight = mask.to(dtype=mse_loss.dtype)
    if weight.shape != mse_loss.shape:
        weight = weight.expand_as(mse_loss)

    weighted_loss = mse_loss * weight
    normalizer = weight.sum().clamp_min(1e-8)
    return weighted_loss.sum() / normalizer

def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class Trainer:
    def __init__(
        self,
        model,
        diffuser,
        data_loader,
        epochs=1000,
        lr=1e-4,
        ema_decay=0.9999,
        device="cuda",
        save_path=None,
        type="e",
        latent="false",
        vae=None,
        vae_scale=None,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.ema_model = deepcopy(model).to(self.device)
        self.ema_model.eval()

        self.use_latent = str(latent).lower() == "true"
        self.vae = None
        self.vae_scale = vae_scale
        if self.use_latent:
            if vae is not None:
                self.vae = vae.to(self.device)
                self.vae_scale = vae_scale if vae_scale is not None else getattr(self.vae, "latent_scale", 1.0)
            else:
                self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
                self.vae_scale = vae_scale if vae_scale is not None else 0.18215
            requires_grad(self.vae, False)
            self.vae.eval()

        self.diffuser = diffuser
        self.data_loader = data_loader
        self.epochs = epochs
        self.lr = lr
        self.ema_decay = ema_decay
        self.save_path = save_path if save_path else './checkpoints/new/model_checkpoint'
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,get_cosine_lambda(initial_lr=self.lr,final_lr=1e-5,epochs=self.epochs,warmup_epoch=200))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.type = type
        self.total_steps = self.epochs * len(self.data_loader)
        self.progress_bar = tqdm(total=self.total_steps, desc="Training", dynamic_ncols=True)      

    def _encode_latents(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("Latent training requested but no VAE was initialised.")

        if hasattr(self.vae, "encode_latents"):
            return self.vae.encode_latents(tensor)

        encoded = self.vae.encode(tensor)
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.sample()
        elif isinstance(encoded, (list, tuple)):
            latents = encoded[0]
        else:
            latents = encoded

        scale = self.vae_scale if self.vae_scale is not None else 1.0
        return latents * scale

    def train_step(self, model: torch.nn.Module, batch):
        
        y_true = batch["diffuse"].float().to(self.device)
        targets = batch["unmasked_residual"].float().to(self.device)
        mask = batch["soft_mask"].float().to(self.device)
        condition = batch["input"].float().to(self.device)


        if self.use_latent:
            condition = self._encode_latents(condition)
            targets = self._encode_latents(targets)
            mask = F.interpolate(mask, size=targets.shape[2:], mode='bilinear', align_corners=False)

        B = condition.size(0)
        t = torch.randint(1, self.diffuser.steps + 1, (B,), dtype=torch.long).to(self.device)
        noise = torch.randn_like(targets).to(self.device)  # shape: [B, 3, H, W]

        if self.type == "weighted":
            noisy_xt = self.diffuser.forward_diffusion(targets, t, noise)
            prediction = model(noisy_xt, t, condition)
            loss = weighted_mse_loss(prediction, noise, mask)
        elif self.type == "unweighted":
            noisy_xt = self.diffuser.forward_diffusion(targets, t, noise)
            prediction = model(noisy_xt, t, condition)
            loss = F.mse_loss(prediction, noise)
        else:
            raise ValueError(f"Unknown training type: {self.type}")
        del batch, condition, targets
        #torch.cuda.empty_cache()
        return loss

    def train(self):
        try:
            self.model.train()
            loss_history = []

            plt.ion()
            # fig, ax = plt.subplots()
            # (line,) = ax.plot([], [], label='Training Loss')  

            # ax.set_xlabel('Epoch')
            # ax.set_ylabel('Loss')
            # ax.set_title('Training Progress')
            # ax.grid(True)
            # ax.legend()

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                
                        
                for step, batch in enumerate(self.data_loader):
                        self.optimizer.zero_grad()
                        
                        loss = self.train_step(self.model, batch)

                        loss.backward()
                        self.optimizer.step()

                        update_ema(self.ema_model, self.model, decay=self.ema_decay)
                        
                        epoch_loss += loss.item()
                        self.progress_bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
                        self.progress_bar.update(1)

                        del batch , loss
                        #torch.cuda.empty_cache()
                        

                loss_history.append(epoch_loss / len(self.data_loader))
                self.scheduler.step()

                # Live plot
                # line.set_xdata(range(len(loss_history)))
                # line.set_ydata(loss_history)
                # ax.relim()
                # ax.autoscale_view()
                # plt.draw()
                # plt.pause(0.01)

                # Save checkpoint
                if (epoch + 1) % 500 == 0:
                    Checkpoint_save(self.model, loss_history[-1], epoch + 1, self.save_path,Parameterization=self.type)

            #plt.ioff()
            #plt.savefig("training_loss_curve.png")
            print("Training complete.")
            
            Checkpoint_save(self.model, loss_history[-1], epoch + 1, self.save_path,Parameterization=self.type)

            return {
                "ema_model": self.ema_model,
                "last_model": self.model
            }

        except Exception as e:
            import traceback
            Error_log(self.model.__class__.__name__, traceback.format_exc())
            
