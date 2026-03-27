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
# from your_model_file import UViT, Diffuser  # <-- Make sure to import your model and diffuser here

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(wait=False): pass  # fallback if not in notebook


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
        objective="e"
    ):
        self.device = device
        self.model = model.to(self.device)
        self.ema_model = deepcopy(model).to(self.device)
        self.ema_model.eval()

        self.diffuser = diffuser
        self.data_loader = data_loader
        self.epochs = epochs
        self.lr = lr
        self.ema_decay = ema_decay
        self.save_path = save_path if save_path else './checkpoints/new/model_checkpoint'
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.type = objective
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,get_cosine_lambda(initial_lr=self.lr,final_lr=1e-5,epochs=self.epochs,warmup_epoch=200))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.total_steps = self.epochs * len(self.data_loader)
        self.progress_bar = tqdm(total=self.total_steps, desc="Training", dynamic_ncols=True)      

    def train_step(self, model: torch.nn.Module, batch):
          # shape: [B, C, H, W]

        targets = batch.get("unmasked_residual")
        if targets is None:
            targets = batch.get("residual")
        if targets is None:
            raise KeyError("Batch must include 'unmasked_residual' or 'residual'.")

        condition = batch.get("input")
        if condition is None:
            condition = batch.get("glossy")
        if condition is None:
            raise KeyError("Batch must include 'input' or 'glossy'.")

        targets = targets.float().to(self.device)
        condition = condition.float().to(self.device)

        B = condition.size(0)
        t = torch.randint(0, self.diffuser.steps, (B,), dtype=torch.long).to(self.device)
        noise = torch.randn_like(targets).to(self.device)  # shape: [B, 3, H, W]

        if self.type == "e":
            noisy_xt = self.diffuser.forward_diffusion(targets, t, noise)
            prediction = model(noisy_xt, t, condition)
            loss = F.mse_loss(prediction, noise)
        elif self.type == "x":
            noisy_xt = self.diffuser.forward_diffusion(targets, t, noise)
            prediction = model(noisy_xt, t, condition)
            loss = F.mse_loss(prediction, targets)
        elif self.type == "v":
            noisy_xt = self.diffuser.forward_diffusion(targets, t, noise)
            velocity = self.diffuser.calculate_velocity(targets, t, noise)
            prediction = model(noisy_xt, t, condition)
            loss = F.mse_loss(prediction, velocity)
        else:
            raise ValueError(f"Unknown training type: {self.type}")
        del batch, condition, targets
        #torch.cuda.empty_cache()
        return loss

    def train(self):
        try:
            self.model.train()
            loss_history = []

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


                # Save checkpoint
                if (epoch + 1) % 200 == 0:
                    Checkpoint_save(self.model, loss_history[-1], epoch + 1, self.save_path,Parameterization=self.type)


            print("Training complete.")
            
            Checkpoint_save(self.model, loss_history[-1], epoch + 1, self.save_path,Parameterization=self.type)

            return {
                "ema_model": self.ema_model,
                "last_model": self.model
            }

        except Exception as e:
            import traceback
            Error_log(self.model.__class__.__name__, traceback.format_exc())
            
