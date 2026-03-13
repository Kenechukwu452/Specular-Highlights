import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Optional, Union


ArrayLike = Union[np.ndarray, torch.Tensor]

def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _otsu_threshold_1d(vals: torch.Tensor, nbins: int = 256, eps: float = 1e-12) -> torch.Tensor:
    """
    Otsu threshold for 1D float tensor vals (assumed in [0,1] but will be clamped).
    Returns a scalar threshold.
    """
    v = vals.detach()
    v = torch.clamp(v, 0.0, 1.0)

    # histogram over [0,1]
    hist = torch.histc(v, bins=nbins, min=0.0, max=1.0)
    hist = hist / (hist.sum() + eps)  # normalised

    # bin centres
    bin_centres = (torch.arange(nbins, device=v.device, dtype=v.dtype) + 0.5) / nbins  # [0,1]

    # cumulative sums
    w0 = torch.cumsum(hist, dim=0)                      # class 0 weight
    w1 = 1.0 - w0                                       # class 1 weight
    mu = torch.cumsum(hist * bin_centres, dim=0)        # cumulative mean
    mu_t = mu[-1]                                       # total mean

    # class means
    mu0 = mu / (w0 + eps)
    mu1 = (mu_t - mu) / (w1 + eps)

    # between-class variance
    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2

    # best threshold index
    k = torch.argmax(sigma_b2)
    return bin_centres[k]


def calculate_threshold(prob_mask: torch.Tensor,
                        q: float = 0.10,
                        aoi: Optional[torch.Tensor] = None,
                        method: str = "quantile",
                        nbins: int = 256) -> torch.Tensor:
    """
    prob_mask: (B,1,H,W), values in [0,1], 0=highlight, 1=non-highlight
    q: quantile used when method="quantile"
    aoi: optional (B,1,H,W) or (B,H,W) boolean (or 0/1) mask M_i
    method: "quantile" or "otsu"
    nbins: histogram bins for Otsu

    Returns:
      p: (B,) per-image thresholds
    """
    if prob_mask.ndim != 4 or prob_mask.shape[1] != 1:
        raise ValueError(f"Expected (B,1,H,W), got {tuple(prob_mask.shape)}")

    method = method.lower()
    if method not in ("quantile", "otsu"):
        raise ValueError(f"method must be 'quantile' or 'otsu', got '{method}'")

    B = prob_mask.shape[0]
    ps = []

    for b in range(B):
        vals = prob_mask[b, 0]  # (H,W)

        if aoi is not None:
            m = aoi[b]
            if m.ndim == 3:  # (1,H,W)
                m = m[0]
            m = m.bool() if m.dtype == torch.bool else (m > 0)
            vals = vals[m]

        vals = vals.flatten()
        if vals.numel() == 0:
            # Fallback to whole image if AoI is empty
            vals = prob_mask[b, 0].flatten()

        if method == "quantile":
            ps.append(torch.quantile(vals, q))
        else:  # otsu
            ps.append(_otsu_threshold_1d(vals, nbins=nbins))

    return torch.stack(ps)  # (B,)





def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    k1 = torch.exp(-0.5 * (coords / sigma) ** 2)
    k1 = k1 / k1.sum()
    k2 = (k1[:, None] * k1[None, :])  # (K,K)
    return k2[None, None, :, :]       # (1,1,K,K)

def apply_gaussian_blur(mask_bchw: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    mask_bchw: (B,1,H,W)
    """
    if mask_bchw.ndim != 4 or mask_bchw.shape[1] != 1:
        raise ValueError(f"Expected (B,1,H,W), got {tuple(mask_bchw.shape)}")
    
    if sigma < 1:
        sigma = sigma + 1e-5  # avoid zero sigma which causes NaNs in kernel


    k = _gaussian_kernel2d(kernel_size, sigma, mask_bchw.device, mask_bchw.dtype)
    pad = kernel_size // 2
    return F.conv2d(mask_bchw, k, padding=pad)




@dataclass
class ProbMaskThresholdOperator:
    p: Union[float, torch.Tensor]  # float or (B,)
    op_above: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    op_below: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    outside_value: float = 1.0

    def __call__(self, prob_mask: torch.Tensor, aoi: Optional[torch.Tensor] = None) -> torch.Tensor:
        if prob_mask.ndim != 4 or prob_mask.shape[1] != 1:
            raise ValueError(f"Expected prob_mask (B,1,H,W), got {tuple(prob_mask.shape)}")

        op_above = self.op_above or (lambda x: x)
        op_below = self.op_below or (lambda x: x)

        # broadcast p to (B,1,1,1)
        if isinstance(self.p, torch.Tensor):
            p = self.p.view(-1, 1, 1, 1).to(device=prob_mask.device, dtype=prob_mask.dtype)
        else:
            p = torch.tensor(float(self.p), device=prob_mask.device, dtype=prob_mask.dtype).view(1, 1, 1, 1)

        out = prob_mask.clone()

        if aoi is None:
            above = prob_mask > p
            out[above] = op_above(prob_mask[above])
            out[~above] = op_below(prob_mask[~above])
            return out

        # --- AoI handling ---
        if aoi.ndim == 4:
            aoi_bool = aoi[:, 0].bool() if aoi.dtype == torch.bool else (aoi[:, 0] > 0)
        elif aoi.ndim == 3:
            aoi_bool = aoi.bool() if aoi.dtype == torch.bool else (aoi > 0)
        else:
            raise ValueError(f"aoi must be (B,1,H,W) or (B,H,W), got {tuple(aoi.shape)}")

        # set outside AoI to 1
        out[:, 0][~aoi_bool] = torch.tensor(self.outside_value, device=out.device, dtype=out.dtype)

        # FIX: reshape p for broadcasting with (B,H,W)
        p2 = p[:, 0, 0, 0].view(-1, 1, 1)  # (B,1,1)

        above_in = (prob_mask[:, 0] > p2) & aoi_bool
        below_in = (prob_mask[:, 0] <= p2) & aoi_bool

        out_vals = out[:, 0]
        in_vals = prob_mask[:, 0]

        out_vals[above_in] = op_above(in_vals[above_in])
        out_vals[below_in] = op_below(in_vals[below_in])

        out[:, 0] = out_vals
        return out

def set_to_one(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)

def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def generate_dynamic_highlight_mask(prob_mask: torch.Tensor,
                                    p: Optional[float] = None,
                                    threshold_method: str = "quantile",
                                    q: float = 0.10,
                                    aoi: Optional[torch.Tensor] = None,
                                    kernel_size: int = 5,
                                    sigma: float = 1.0) -> torch.Tensor:
    """
    Returns soft_mask: (B,1,H,W)
    - If p is provided: uses fixed threshold p
    - Else: computes per-image p via quantile q
    """
    if prob_mask.ndim != 4 or prob_mask.shape[1] != 1:
        prob_mask = prob_mask.unsqueeze(1)  # try to add channel dim if missing
    
    if p is None:
        p_b = calculate_threshold(prob_mask, method=threshold_method, q=q, aoi=aoi)  # (B,)
        op = ProbMaskThresholdOperator(p=p_b, op_above=set_to_one, op_below=identity)
    else:
        op = ProbMaskThresholdOperator(p=float(p), op_above=set_to_one, op_below=identity)

    muted = op(prob_mask)  # (B,1,H,W), values > p forced to 1
    soft = apply_gaussian_blur(muted, kernel_size=kernel_size, sigma=sigma)
    
    return soft

