import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from pipeline.ResidualLoader import compute_residual


valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def apply_paired_geometry_augment(
    glossy_img: torch.Tensor,
    diffuse_img: torch.Tensor,
    random_hflip_p: float = 0.5,
    random_vflip_p: float = 0.0,
    enable_rot90: bool = False,
):
    if torch.rand(1).item() < random_hflip_p:
        glossy_img = TF.hflip(glossy_img)
        diffuse_img = TF.hflip(diffuse_img)

    if torch.rand(1).item() < random_vflip_p:
        glossy_img = TF.vflip(glossy_img)
        diffuse_img = TF.vflip(diffuse_img)

    if enable_rot90:
        k = int(torch.randint(0, 4, (1,)).item())
        glossy_img = torch.rot90(glossy_img, k=k, dims=(-2, -1))
        diffuse_img = torch.rot90(diffuse_img, k=k, dims=(-2, -1))

    return glossy_img, diffuse_img


class PSDDiffusionDataset(Dataset):
    def __init__(
        self,
        diffuse_dir,
        glossy_dir,
        resize_hw=None,
        residual_mode="additive",
        augment=False,
        random_hflip_p=0.5,
        random_vflip_p=0.0,
        enable_rot90=False,
        paired_transform=None,
    ):
        self.diffuse_dir = Path(diffuse_dir)
        self.glossy_dir = Path(glossy_dir)
        if resize_hw is None:
            self.resize_hw = None
        elif isinstance(resize_hw, int):
            self.resize_hw = (resize_hw, resize_hw)
        else:
            if len(resize_hw) != 2:
                raise ValueError(f"resize_hw must be an int or a length-2 sequence, got: {resize_hw}")
            self.resize_hw = tuple(int(dim) for dim in resize_hw)

        if residual_mode not in {"additive", "subtractive"}:
            raise ValueError(f"Invalid residual_mode: {residual_mode}")

        self.residual_mode = residual_mode
        self.augment = augment
        self.random_hflip_p = float(random_hflip_p)
        self.random_vflip_p = float(random_vflip_p)
        self.enable_rot90 = bool(enable_rot90)
        self.paired_transform = paired_transform

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
        if self.resize_hw is not None:
            img = TF.resize(
                img,
                self.resize_hw,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        return TF.to_tensor(img)

    def _apply_transforms(self, glossy_img: torch.Tensor, diffuse_img: torch.Tensor):
        if self.paired_transform is not None:
            return self.paired_transform(glossy_img, diffuse_img)
        if not self.augment:
            return glossy_img, diffuse_img
        return apply_paired_geometry_augment(
            glossy_img,
            diffuse_img,
            random_hflip_p=self.random_hflip_p,
            random_vflip_p=self.random_vflip_p,
            enable_rot90=self.enable_rot90,
        )

    def __getitem__(self, idx):
        glossy_path, diffuse_path, name = self.samples[idx]

        glossy_img = self._load_rgb(glossy_path)
        diffuse_img = self._load_rgb(diffuse_path)
        glossy_img, diffuse_img = self._apply_transforms(glossy_img, diffuse_img)

        residual = compute_residual(
            glossy_img,
            diffuse_img,
            residual_mode=self.residual_mode,
        )

        return {
            "glossy": glossy_img,
            "diffuse": diffuse_img,
            "residual": residual,
            "name": name,
        }


def make_diffusion_dataloader(
    diffuse_dir,
    glossy_dir,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    resize_hw=None,
    residual_mode="additive",
    augment=False,
    random_hflip_p=0.5,
    random_vflip_p=0.0,
    enable_rot90=False,
    paired_transform=None,
):
    dataset = PSDDiffusionDataset(
        diffuse_dir=diffuse_dir,
        glossy_dir=glossy_dir,
        resize_hw=resize_hw,
        residual_mode=residual_mode,
        augment=augment,
        random_hflip_p=random_hflip_p,
        random_vflip_p=random_vflip_p,
        enable_rot90=enable_rot90,
        paired_transform=paired_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader
