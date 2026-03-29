
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
        """
        This needs some work
            1. Betas not preped for this yet
            2. Wrong formula 
        """
        velocity = self.alphas_bar[t]*noise -self.one_minus_alphas_bar[t]*x0

        return velocity

    def sample_from_noise(self, model, condition, show_progress=True, ddim=False, skip_steps= 2, v_parm = False):
        with torch.no_grad():
            x_t = torch.randn_like(condition)
            t_now = torch.tensor([self.steps], device=self.device).repeat(x_t.shape[0])
            t_pre = t_now - (skip_steps if ddim else 1)  
            ddim_bar  = range(0, self.steps, skip_steps)
            ddpm_bar = range(self.steps)
            if show_progress:
                if ddim:
                    p_bar = tqdm(ddim_bar)  
                else:
                    p_bar = tqdm(ddpm_bar)  
            else:
                if ddim:
                    p_bar = ddim_bar
                else:
                    p_bar = ddpm_bar
            
            for t in p_bar:
                
                predicted_noise = model(x_t, t_now, condition)
                
                
                if ddim:
                    x_t, x_0 = self.DDIM_sample_step(x_t, t_now, t_pre, predicted_noise) if v_parm ==False else self.DDIM_Velocity_sample_step(x_t, t_now, t_pre, predicted_noise)

                    # Handle final steps for DDIM
                    if t == ddim_bar[-1]:
                        return x_0 
    
                else:
                    x_t = self.DDPM_sample_step(x_t, t_now, t_pre, predicted_noise)
                
                t_now = t_pre
                t_pre = t_pre - (skip_steps if ddim else 1)
            
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
        '''
                Fix the velocity modified DDIm sampling function
        '''
        coef1 = self.sqrt_alphas_bar[t_pre]
        coef2 = self.sqrt_one_minus_alphas_bar[t]
        coef3 = 1/self.sqrt_alphas_bar[t]
        #sig = stochacity * ( torch.sqrt(self.one_minus_alphas[t_pre]/self.one_minus_alphas[t]) *  torch.sqrt(self.one_minus_alphas[t]/self.alphas[t_pre]))
        #sig_sqr = torch.square(sig)
        coef4 = self.sqrt_one_minus_alphas_bar[t_pre] #+sig_sqr)
        x_0_pred = coef3 * (x_t-coef2*velocity)
        x_t_dir = (coef4*velocity)
        x_t_pre = coef1*x_0_pred + x_t_dir  #+ sig*torch.randn_like(x_t)
        return  x_t_pre, x_0_pred

        
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
        self.betas = self.betas.view(self.steps+1, 1, 1, 1)
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
