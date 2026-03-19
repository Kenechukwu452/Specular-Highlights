from collections import deque
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from pipeline import dhl_generator as dhl_gen


def compute_residual(glossy_img, diffuse_img, residual_mode: str = "additive"):
    # additive  : diffuse = glossy + residual
    # subtractive: diffuse = glossy - residual
    if residual_mode == "additive":
        return diffuse_img - glossy_img
    if residual_mode == "subtractive":
        return glossy_img - diffuse_img
    raise ValueError(f"Invalid residual_mode: {residual_mode}")


def reconstruct_diffuse(glossy_img, residual, residual_mode: str = "additive"):
    if residual_mode == "additive":
        return glossy_img + residual
    if residual_mode == "subtractive":
        return glossy_img - residual
    raise ValueError(f"Invalid residual_mode: {residual_mode}")


class PSDResidualDataset(Dataset):
    def __init__(
        self,
        diffuse_dir,
        glossy_dir,
        threshold_method="otsu",
        threshold=0.95,
        kernel_size=3,
        soft_gamma=0.5,
        fill_holes=True,
        residual_mode="additive",
    ):
        self.diffuse_dir = Path(diffuse_dir)
        self.glossy_dir = Path(glossy_dir)
        self.threshold_method = threshold_method
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.soft_gamma = soft_gamma
        self.fill_holes = fill_holes
        self.residual_mode = residual_mode

        if self.threshold_method not in {"fixed", "otsu", "quantile"}:
            raise ValueError(f"Invalid threshold_method: {self.threshold_method}")

        if self.residual_mode not in {"additive", "subtractive"}:
            raise ValueError(f"Invalid residual_mode: {self.residual_mode}")

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

        diffuse_files = {
            p.name: p for p in self.diffuse_dir.iterdir()
            if p.suffix.lower() in valid_exts
        }
        glossy_files = {
            p.name: p for p in self.glossy_dir.iterdir()
            if p.suffix.lower() in valid_exts
        }

        common_names = sorted(set(diffuse_files.keys()) & set(glossy_files.keys()))
        if len(common_names) == 0:
            raise ValueError("No matching image names found.")

        self.samples = [
            (glossy_files[name], diffuse_files[name], name)
            for name in common_names
        ]

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img)

    def _morph_open_close(self, mask: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size
        if k <= 1:
            return mask

        pad = k // 2
        x = mask.unsqueeze(0).unsqueeze(0)

        eroded = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)
        opened = F.max_pool2d(eroded, kernel_size=k, stride=1, padding=pad)

        dilated = F.max_pool2d(opened, kernel_size=k, stride=1, padding=pad)
        closed = -F.max_pool2d(-dilated, kernel_size=k, stride=1, padding=pad)

        return closed.squeeze(0).squeeze(0)

    def _compute_threshold(self, soft_mask: torch.Tensor) -> torch.Tensor:
        prob_mask = soft_mask.unsqueeze(0).unsqueeze(0)

        if self.threshold_method == "fixed":
            return torch.tensor(float(self.threshold), device=soft_mask.device, dtype=soft_mask.dtype)

        thresholds = dhl_gen.calculate_threshold(
            prob_mask,
            method=self.threshold_method,
            q=self.threshold,
        )
        return thresholds[0]

    def _fill_mask_holes(self, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = (mask > 0.5).detach().cpu()
        height, width = mask_bool.shape

        if height == 0 or width == 0:
            return mask

        background = ~mask_bool
        border_background = torch.zeros_like(background, dtype=torch.bool)
        queue = deque()

        def visit(row: int, col: int) -> None:
            if not background[row, col] or border_background[row, col]:
                return
            border_background[row, col] = True
            queue.append((row, col))

        for col in range(width):
            visit(0, col)
            visit(height - 1, col)

        for row in range(height):
            visit(row, 0)
            visit(row, width - 1)

        neighbors = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )

        while queue:
            row, col = queue.popleft()
            for d_row, d_col in neighbors:
                next_row = row + d_row
                next_col = col + d_col
                if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                    continue
                visit(next_row, next_col)

        holes = background & ~border_background
        filled_mask = mask_bool | holes
        return filled_mask.to(device=mask.device, dtype=mask.dtype)

    def _extract_mask(self, glossy_img, diffuse_img):
        # Use residual magnitude itself as the object/residual confidence map.
        residual_energy = (glossy_img - diffuse_img).abs().mean(dim=0)

        soft_mask = residual_energy / (residual_energy.max() + 1e-8)
        soft_mask = torch.pow(soft_mask, self.soft_gamma)
        soft_mask = soft_mask.clamp(0.0, 1.0)

        threshold = self._compute_threshold(soft_mask)
        otsu_mask = (soft_mask > threshold).float()
        otsu_mask = self._morph_open_close(otsu_mask)
        if self.fill_holes:
            otsu_mask = self._fill_mask_holes(otsu_mask)
        otsu_mask = otsu_mask.clamp(0.0, 1.0)

        return soft_mask.unsqueeze(0).float(), otsu_mask.unsqueeze(0).float()

    def __getitem__(self, idx):
        glossy_path, diffuse_path, name = self.samples[idx]

        glossy_img = self._load_rgb(glossy_path)
        diffuse_img = self._load_rgb(diffuse_path)

        soft_mask, mask = self._extract_mask(glossy_img, diffuse_img)

        unmasked_additive_residual = compute_residual(
            glossy_img,
            diffuse_img,
            residual_mode="additive",
        )
        unmasked_subtractive_residual = compute_residual(
            glossy_img,
            diffuse_img,
            residual_mode="subtractive",
        )
        unmasked_target_residual = compute_residual(
            glossy_img,
            diffuse_img,
            residual_mode=self.residual_mode,
        )

        # Keep the Otsu-masked residual outputs as well for object-only correction.
        residual_mask = mask.expand_as(unmasked_target_residual)
        masked_additive_residual = unmasked_additive_residual * residual_mask
        masked_subtractive_residual = unmasked_subtractive_residual * residual_mask
        masked_target_residual = unmasked_target_residual * residual_mask

        return {
            "input": glossy_img,
            # Backward-compatible masked outputs.
            "target": masked_target_residual,
            "residual": masked_target_residual,
            "masked_target": masked_target_residual,
            "masked_residual": masked_target_residual,
            "diffuse": diffuse_img,
            "soft_mask": soft_mask,
            "mask": mask,
            "additive_residual": masked_additive_residual,
            "subtractive_residual": masked_subtractive_residual,
            "masked_additive_residual": masked_additive_residual,
            "masked_subtractive_residual": masked_subtractive_residual,
            # Full residuals are also returned so you can use the original values.
            "unmasked_target": unmasked_target_residual,
            "unmasked_residual": unmasked_target_residual,
            "unmasked_additive_residual": unmasked_additive_residual,
            "unmasked_subtractive_residual": unmasked_subtractive_residual,
            "name": name,
        }


def make_residual_dataloader(
    diffuse_dir,
    glossy_dir,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    threshold_method="otsu",
    threshold=0.95,
    kernel_size=3,
    soft_gamma=0.5,
    fill_holes=True,
    residual_mode="additive",
):
    dataset = PSDResidualDataset(
        diffuse_dir=diffuse_dir,
        glossy_dir=glossy_dir,
        threshold_method=threshold_method,
        threshold=threshold,
        kernel_size=kernel_size,
        soft_gamma=soft_gamma,
        fill_holes=fill_holes,
        residual_mode=residual_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader
