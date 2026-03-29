import numpy as np
import torch
from tqdm import tqdm


class Diffuser:
    def __init__(self, steps, device):
        self.device = torch.device(device)
        self.steps = int(steps)
        self.beta_source = torch.empty(self.steps, dtype=torch.float32)
        self.betas = torch.empty(0, dtype=torch.float32)
        self.alphas = torch.empty(0, dtype=torch.float32)
        self.alphas_bar = torch.empty(0, dtype=torch.float32)
        self.alphas_bar_prev = torch.empty(0, dtype=torch.float32)
        self.sqrt_alphas = torch.empty(0, dtype=torch.float32)
        self.sqrt_alphas_bar = torch.empty(0, dtype=torch.float32)
        self.sqrt_one_minus_alphas_bar = torch.empty(0, dtype=torch.float32)
        self.posterior_variance = torch.empty(0, dtype=torch.float32)
        self.posterior_mean_coef1 = torch.empty(0, dtype=torch.float32)
        self.posterior_mean_coef2 = torch.empty(0, dtype=torch.float32)

    def _normalize_parameterization(self, parameterization: str) -> str:
        mapping = {
            "e": "e",
            "eps": "e",
            "epsilon": "e",
            "x": "x",
            "x0": "x",
            "v": "v",
            "velocity": "v",
        }
        try:
            return mapping[parameterization.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported parameterization: {parameterization}") from exc

    def change_device(self, device):
        self.device = torch.device(device)
        self._generate_parameters_from_beta()

    def generate_parameters_from_beta(self):
        self._generate_parameters_from_beta()

    def _generate_parameters_from_beta(self):
        self.betas = self.beta_source.to(self.device, dtype=torch.float32).view(self.steps, 1, 1, 1)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = torch.cat([torch.ones_like(self.alphas_bar[:1]), self.alphas_bar[:-1]], dim=0)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt((1.0 - self.alphas_bar).clamp_min(0.0))

        denom = (1.0 - self.alphas_bar).clamp_min(1e-12)
        self.posterior_variance = self.betas * (1.0 - self.alphas_bar_prev) / denom
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_bar_prev) / denom
        self.posterior_mean_coef2 = torch.sqrt(self.alphas) * (1.0 - self.alphas_bar_prev) / denom

    def forward_diffusion(self, x0, t, noise):
        return self.sqrt_alphas_bar[t] * x0 + self.sqrt_one_minus_alphas_bar[t] * noise

    def calculate_velocity(self, x0, t, noise):
        alpha = self.sqrt_alphas_bar[t]
        sigma = self.sqrt_one_minus_alphas_bar[t]
        return alpha * noise - sigma * x0

    def training_target(self, x0, t, noise, parameterization="e"):
        parameterization = self._normalize_parameterization(parameterization)
        if parameterization == "e":
            return noise
        if parameterization == "x":
            return x0
        return self.calculate_velocity(x0, t, noise)

    def predict_x0_and_noise(self, x_t, t, model_output, parameterization="e"):
        parameterization = self._normalize_parameterization(parameterization)
        alpha = self.sqrt_alphas_bar[t]
        sigma = self.sqrt_one_minus_alphas_bar[t]

        if parameterization == "e":
            pred_noise = model_output
            pred_x0 = (x_t - sigma * pred_noise) / alpha.clamp_min(1e-12)
            return pred_x0, pred_noise

        if parameterization == "x":
            pred_x0 = model_output
            pred_noise = (x_t - alpha * pred_x0) / sigma.clamp_min(1e-12)
            return pred_x0, pred_noise

        pred_velocity = model_output
        pred_x0 = alpha * x_t - sigma * pred_velocity
        pred_noise = sigma * x_t + alpha * pred_velocity
        return pred_x0, pred_noise

    def q_posterior(self, x0, x_t, t):
        mean = self.posterior_mean_coef1[t] * x0 + self.posterior_mean_coef2[t] * x_t
        variance = self.posterior_variance[t]
        return mean, variance

    def ddpm_sample_step(self, x_t, t, model_output, parameterization="e"):
        pred_x0, _ = self.predict_x0_and_noise(x_t, t, model_output, parameterization=parameterization)
        mean, variance = self.q_posterior(pred_x0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).view(-1, 1, 1, 1).to(x_t.dtype)
        return mean + nonzero_mask * torch.sqrt(variance.clamp_min(1e-20)) * noise, pred_x0

    def sample_from_noise(
        self,
        model,
        condition,
        parameterization="e",
        show_progress=True,
        initial_noise=None,
    ):
        parameterization = self._normalize_parameterization(parameterization)
        with torch.no_grad():
            if initial_noise is None:
                x_t = torch.randn_like(condition)
            else:
                x_t = initial_noise.to(condition.device, dtype=condition.dtype).clone()

            iterator = range(self.steps - 1, -1, -1)
            if show_progress:
                iterator = tqdm(iterator)

            for step in iterator:
                t = torch.full((x_t.shape[0],), step, dtype=torch.long, device=condition.device)
                model_output = model(x_t, t, condition)
                x_t, pred_x0 = self.ddpm_sample_step(x_t, t, model_output, parameterization=parameterization)
                if step == 0:
                    return pred_x0

            return x_t


class LinearDiffuser(Diffuser):
    def __init__(self, steps, beta_min, beta_max, device):
        super().__init__(steps, device)
        self.name = "LinearDiffuser"
        self.beta_source = torch.linspace(beta_min, beta_max, steps, dtype=torch.float32)
        self.generate_parameters_from_beta()


class CosSchDiffuser(Diffuser):
    def __init__(self, steps, device):
        super().__init__(steps, device)
        self.name = "CosSchDiffuser"
        s = 0.008
        tlist = torch.arange(1, self.steps + 1, dtype=torch.float32)
        temp1 = torch.cos((tlist / self.steps + s) / (1 + s) * np.pi / 2) ** 2
        temp2 = torch.cos(((tlist - 1) / self.steps + s) / (1 + s) * np.pi / 2) ** 2
        self.beta_source = (1 - (temp1 / temp2)).clamp(max=0.999)
        self.generate_parameters_from_beta()
