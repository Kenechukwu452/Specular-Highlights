#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from Trainer import Trainer as BaseTrainer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import ResidualLoader as residual_loader


class Trainer(BaseTrainer):
    def __init__(
        self,
        *args,
        residual_mode: str = "subtractive",
        use_roi_mask: bool = True,
        use_otsu_mask: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.residual_mode = residual_mode
        if use_otsu_mask is not None:
            use_roi_mask = use_otsu_mask
        self.use_roi_mask = use_roi_mask

    def _masked_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        roi_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_mask = roi_mask.expand_as(prediction)
        squared_error = (prediction - target) ** 2
        masked_squared_error = squared_error * expanded_mask
        denom = expanded_mask.sum().clamp_min(1.0)
        return masked_squared_error.sum() / denom

    def train_step(self, model: torch.nn.Module, batch):
        # Expected batch layout:
        #   0:3 -> glossy input
        #   3:6 -> diffuse target
        #   6:7 -> object-selection ROI mask
        if batch.size(1) < 6:
            raise ValueError(
                "Trainer_v2 expects at least 6 channels per batch: "
                "3 glossy + 3 diffuse."
            )

        condition = batch[:, :3, :, :].to(self.device)
        diffuse = batch[:, 3:6, :, :].to(self.device)

        if self.use_roi_mask:
            if batch.size(1) < 7:
                raise ValueError(
                    "Trainer_v2 expected a 7th channel for the object-selection ROI mask."
                )
            roi_mask = batch[:, 6:7, :, :].to(self.device)
        else:
            roi_mask = torch.ones_like(condition[:, :1, :, :])

        residual = residual_loader.compute_residual(
            condition,
            diffuse,
            residual_mode=self.residual_mode,
        )
        targets = residual * roi_mask

        targets = targets * self.target_scale

        batch_size = condition.size(0)
        t = torch.randint(0, self.diffuser.steps, (batch_size,), dtype=torch.long, device=self.device)
        noise = torch.randn_like(targets)
        noise = noise * roi_mask
        noisy_xt = self.diffuser.forward_diffusion(targets, t, noise)
        prediction = model(noisy_xt, t, condition)
        masked_prediction = prediction * roi_mask

        if self.type == "e":
            loss = self._masked_mse(masked_prediction, noise, roi_mask)
        elif self.type == "x":
            loss = self._masked_mse(masked_prediction, targets, roi_mask)
        else:
            raise ValueError(f"Unknown training type: {self.type}")

        del batch, condition, diffuse, residual, targets, noisy_xt, prediction, masked_prediction, noise, roi_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss
