
#usr/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Diffuser():
    def __init__(self, steps, device):
        self.device = device
        self.steps = steps
        self.betas = torch.tensor([])
        self.beta_source = torch.tensor([])
        self.alphas = 1-self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.one_minus_alphas_bar = 1 - self.alphas_bar
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)
        
    def forward_diffusion(self, x0, t, noise):
        xt = self.sqrt_alphas_bar[t]*x0 + self.sqrt_one_minus_alphas_bar[t]*noise
        return xt
    
    def calculate_velocity(self, x0, t, noise):
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
        velocity = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x0
        return velocity

    def predict_x0_from_noise(self, x_t, t, noise):
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t].clamp_min(1e-12)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
        return (x_t - sqrt_one_minus_alpha_bar_t * noise) / sqrt_alpha_bar_t

    def predict_x0_from_velocity(self, x_t, t, velocity):
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
        return sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * velocity

    def predict_noise_from_x0(self, x_t, t, x0):
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t].clamp_min(1e-12)
        return (x_t - sqrt_alpha_bar_t * x0) / sqrt_one_minus_alpha_bar_t

    def predict_noise_from_velocity(self, x_t, t, velocity):
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
        return sqrt_one_minus_alpha_bar_t * x_t + sqrt_alpha_bar_t * velocity

    def _resolve_prediction(self, x_t, t, model_output, prediction_type):
        if prediction_type == "e":
            predicted_noise = model_output
            x_0_pred = self.predict_x0_from_noise(x_t, t, predicted_noise)
        elif prediction_type == "x":
            x_0_pred = model_output
            predicted_noise = self.predict_noise_from_x0(x_t, t, x_0_pred)
        elif prediction_type == "v":
            x_0_pred = self.predict_x0_from_velocity(x_t, t, model_output)
            predicted_noise = self.predict_noise_from_velocity(x_t, t, model_output)
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
        return predicted_noise, x_0_pred

    def sample_from_noise(self, model, condition, show_progress=True, ddim=False, skip_steps= 2, v_parm = False, prediction_type=None):
        with torch.no_grad():
            x_t = torch.randn_like(condition)
            if prediction_type is None:
                prediction_type = "v" if v_parm else "e"

            if ddim:
                timesteps = list(range(self.steps - 1, -1, -skip_steps))
                if timesteps[-1] != 0:
                    timesteps.append(0)
            else:
                timesteps = list(range(self.steps - 1, -1, -1))

            if show_progress:
                p_bar = tqdm(timesteps)
            else:
                p_bar = timesteps
            
            for idx, current_t in enumerate(p_bar):
                t_now = torch.full((x_t.shape[0],), current_t, device=self.device, dtype=torch.long)
                model_output = model(x_t, t_now, condition)
                predicted_noise, x_0 = self._resolve_prediction(
                    x_t,
                    t_now,
                    model_output,
                    prediction_type=prediction_type,
                )

                if current_t == 0:
                    return x_0

                prev_t = timesteps[idx + 1]
                t_pre = torch.full((x_t.shape[0],), prev_t, device=self.device, dtype=torch.long)

                if ddim:
                    x_t, _ = self.DDIM_sample_step(x_t, t_now, t_pre, predicted_noise)
                else:
                    x_t = self.DDPM_sample_step(x_t, t_now, t_pre, predicted_noise)
            
            return x_t

    def DDPM_sample_step(self, x_t, t, t_pre, noise):
        coef1 = 1/self.sqrt_alphas[t]
        coef2 = self.betas[t]/self.sqrt_one_minus_alphas_bar[t]
        sig = torch.sqrt(self.betas[t])*self.sqrt_one_minus_alphas_bar[t_pre]/self.sqrt_one_minus_alphas_bar[t]
        return coef1*(x_t-coef2*noise)+sig*torch.randn_like(x_t)

    def DDIM_sample_step(self, x_t,t, t_pre, noise):
        coef1 = self.sqrt_alphas_bar[t_pre]
        coef2 = self.sqrt_one_minus_alphas_bar[t]
        coef3 = 1/self.sqrt_alphas_bar[t]
        #sig = stochacity * ( torch.sqrt(self.one_minus_alphas[t_pre]/self.one_minus_alphas[t]) *  torch.sqrt(self.one_minus_alphas[t]/self.alphas[t_pre]))
        #sig_sqr = torch.square(sig)
        coef4 = self.sqrt_one_minus_alphas_bar[t_pre] #+sig_sqr)
        x_0_pred = coef3 * (x_t-coef2*noise)
        x_t_dir = (coef4*noise)
        x_t_pre = coef1*x_0_pred + x_t_dir  #+ sig*torch.randn_like(x_t)
        return  x_t_pre, x_0_pred
    
    def DDIM_Velocity_sample_step(self, x_t,t, t_pre, velocity):
        x_0_pred = self.predict_x0_from_velocity(x_t, t, velocity)
        predicted_noise = self.predict_noise_from_velocity(x_t, t, velocity)
        x_t_pre = self.sqrt_alphas_bar[t_pre] * x_0_pred + self.sqrt_one_minus_alphas_bar[t_pre] * predicted_noise
        return x_t_pre, x_0_pred

        
    def change_device(self, device):
        self.device = device
        self._generate_parameters_from_beta()

    def generate_parameters_from_beta(self):
        self._generate_parameters_from_beta()
        #print('The sqrt_alphas_bar at the last step is {} , be careful if this value is not close to 0!'.format(
        #    self.sqrt_alphas_bar[-1].item()))

    def _generate_parameters_from_beta(self):
        self.betas = torch.cat(
            (torch.tensor([0]), self.beta_source), dim=0) 
        self.betas = self.betas.reshape(self.steps+1, 1, 1, 1)
        self.betas = self.betas.to(self.device)

        self.alphas = 1-self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.one_minus_alphas_bar = 1 - self.alphas_bar
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)



class LinearDiffuser(Diffuser):
    def __init__(self, steps, beta_min, beta_max, device):
        super().__init__(steps, device)
        self.name = "LinearDiffuser"
        self.beta_source = torch.linspace(0, 1, steps) * (beta_max - beta_min) + beta_min
        self.generate_parameters_from_beta()


class CosSchDiffuser(Diffuser):

    def __init__(self, steps, device):
        super().__init__(steps, device)
        self.name = "CosSchDiffuser"
        s = 0.008
        tlist = torch.arange(1, self.steps+1, 1)
        temp1 = torch.cos((tlist/self.steps+s)/(1+s)*np.pi/2)
        temp1 = temp1*temp1
        temp2 = np.cos(((tlist-1)/self.steps+s)/(1+s)*np.pi/2)
        temp2 = temp2*temp2
        self.beta_source = 1-(temp1/temp2)
        self.beta_source[self.beta_source > 0.999] = 0.999
        self.generate_parameters_from_beta()
