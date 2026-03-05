import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Union



ArrayLike = Union[np.ndarray, torch.Tensor]


def _is_torch(x: ArrayLike) -> bool: # Not yet sure if the gradient will be needed,  by other patrts maybe the Pconv
    return isinstance(x, torch.Tensor)


def calculate_threshold(prob_mask: ArrayLike) -> float:
    """
    Calculate a threshold p  
    current logic based on the mean of the probability mask.
    probably wrong replace with right logic when I figure out what it is
    """
    if _is_torch(prob_mask):
        return prob_mask.mean().item()
     
    return prob_mask.mean()

def apply_gaussian_blur(soft_mask: ArrayLike, kernel_size: int = 5, sigma: float = 1.0) -> ArrayLike:
    if _is_torch(soft_mask):
        
        # Create Gaussian kernel for tourch
        def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
            x = torch.arange(-size // 2 + 1., size // 2 + 1.)
            kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d /= kernel_1d.sum()
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            return kernel_2d

        kernel = gaussian_kernel(kernel_size, sigma).to(soft_mask.device).unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        soft_mask = F.conv2d(soft_mask.unsqueeze(0).unsqueeze(0), kernel, padding=padding).squeeze()
    else:
        from scipy.ndimage import gaussian_filter
        soft_mask = gaussian_filter(soft_mask, sigma=sigma)

    return soft_mask


@dataclass
class ProbMaskThresholdOperator:
    p: float 
    op_above: Optional[Callable[[ArrayLike], ArrayLike]] = None
    op_below: Optional[Callable[[ArrayLike], ArrayLike]] = None

    def __call__(self, prob_mask: ArrayLike) -> ArrayLike:
        prob_mask = prob_mask

        op_above = self.op_above or (lambda x: x)
        op_below = self.op_below or (lambda x: x)

        above = prob_mask > self.p
        out = prob_mask.clone() if _is_torch(prob_mask) else prob_mask.copy()

        out[above] = op_above(prob_mask[above])
        out[~above] = op_below(prob_mask[~above])
        return out

# ----- example ops -----

def set_to_one(x: ArrayLike) -> ArrayLike:
    if _is_torch(x):
        return torch.ones_like(x)
    return np.ones_like(x)


def scale(factor: float):
    return lambda x: x * factor


# ----- usage -----

def generate_dynamic_highlight_mask(prob_mask: ArrayLike) -> ArrayLike:
    p = calculate_threshold(prob_mask) 
    operator = ProbMaskThresholdOperator(
        p=p,
        op_above=set_to_one,   # set probabilities above threshold to 1
        op_below=scale(1.0),   # keep the rest unchanged
    )
    muted_mask = operator(prob_mask)
    soft_mask = apply_gaussian_blur(muted_mask, kernel_size=5, sigma=1.0)
    return soft_mask