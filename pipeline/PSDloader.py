import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from pipeline import dhl_generator as dhl_gen


def dhlgenerator( hl = None, blurstrength: float = 1.0, threshold_method: str= "quantile",q: float = 0.10, aoi = None, kernel_size: int = 5 ):
    torch.manual_seed(0)
    prob_mask = hl
    if aoi is not None:
        if hl.shape != aoi.shape:
            raise ValueError(f"hl and aoi must have the same shape, got {hl.shape} and {aoi.shape}")
        
    soft_with_aoi = dhl_gen.generate_dynamic_highlight_mask(prob_mask, threshold_method=threshold_method, q=q, aoi=aoi, kernel_size=kernel_size, sigma=blurstrength)

    return soft_with_aoi



class PSDHighlightDataset(Dataset):
    def __init__(
        self,
        diffuse_dir,
        specular_dir,
        threshold_method="quantile",   # "fixed", "otsu", "quantile"
        threshold=0.95,             # threshold on soft_mask
                             # quantile on soft_mask
        kernel_size=3,
        soft_gamma=0.5,
    ):
        self.diffuse_dir = Path(diffuse_dir)
        self.specular_dir = Path(specular_dir)
        self.threshold_method = threshold_method
        
        self.kernel_size = kernel_size
        self.soft_gamma = soft_gamma

        
        if self.threshold_method == "otsu":
            self.threshold = threshold
        elif self.threshold_method == "quantile":
           self.threshold = threshold
        else:
            raise ValueError(f"Invalid threshold_method: {self.threshold_method}")

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

        diffuse_files = {
            p.name: p for p in self.diffuse_dir.iterdir()
            if p.suffix.lower() in valid_exts
        }
        specular_files = {
            p.name: p for p in self.specular_dir.iterdir()
            if p.suffix.lower() in valid_exts
        }

        common_names = sorted(set(diffuse_files.keys()) & set(specular_files.keys()))
        if len(common_names) == 0:
            raise ValueError("No matching image names found.")

        self.samples = [
            (specular_files[name], diffuse_files[name], name)
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

    def _extract_mask(self, img_spec, img_diff):
        # positive difference only
        diff = torch.clamp(img_spec - img_diff, min=0.0)
        diff_map = diff.mean(dim=0)

        # soft mask:
        # 0 = highlight, 1 = non-highlight
        soft_mask = diff_map / (diff_map.max() + 1e-8)
        soft_mask = torch.pow(soft_mask, self.soft_gamma)
        soft_mask = 1.0 - soft_mask
        soft_mask = soft_mask.clamp(0.0, 1.0)
        #apply dynamic highlight mask generator
        dhlmask  = dhlgenerator(hl =soft_mask.unsqueeze(0).unsqueeze(0), blurstrength=0.4, threshold_method=self.threshold_method, aoi=None, q = self.threshold, kernel_size=self.kernel_size)
        dhlmask = dhlmask[0, 0].cpu()
        dhlmask =dhlmask.clamp(0.0, 1.0)

        return soft_mask.unsqueeze(0).float(), dhlmask.unsqueeze(0).float()

    def __getitem__(self, idx):
        spec_path, diff_path, name = self.samples[idx]

        spec_img = self._load_rgb(spec_path)
        diff_img = self._load_rgb(diff_path)

        soft_mask, mask = self._extract_mask(spec_img, diff_img)

        return {
            "input": spec_img,
            "target": diff_img,
            "soft_mask": soft_mask,   # continuous, 0=highlight, 1=non-highlight
            "mask": mask,             # thresholded soft mask, same convention
            "name": name,
        }


def make_psd_dataloader(
    diffuse_dir,
    specular_dir,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    threshold_method="fixed",
    threshold=0.95,
    kernel_size=3,
    soft_gamma=0.5,
):
    dataset = PSDHighlightDataset(
        diffuse_dir=diffuse_dir,
        specular_dir=specular_dir,
        threshold_method=threshold_method,
        threshold=threshold,
        kernel_size=kernel_size,
        soft_gamma=soft_gamma,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader