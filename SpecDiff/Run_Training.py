#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

import Backbone
import Diffuser as diff
import SDEBackbone
import SingBackbone
import Trainer as T


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PIL_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
SPLIT_DIRS = {
    "train": ("PSD_Train/PSD_Train_specular", "PSD_Train/PSD_Train_diffuse"),
    "val": ("PSD_val/PSD_val_specular", "PSD_val/PSD_val_diffuse"),
    "test": ("PSD_Test/PSD_Test_specular", "PSD_Test/PSD_Test_diffuse"),
}

# Edit these settings and run this file directly.
RUN_CONFIG = argparse.Namespace(
    dataset_source="psd",  # "psd" or "local"
    psd_root="/share/lcn_projects/z0058vfs/project_hl.PSD_Dataset",
    split="train",
    glossy_dir=None,
    diffuse_dir=None,
    backbone_module="SDEBackbone",  # "SDEBackbone", "Backbone", "SingBackbone"
    model="unetwithtransformer",
    image_size=32,
    batch_size=10,
    noise_steps=200,
    epochs=200,
    lr=1e-4,
    depth=4,
    parameterization="e",
    target_scale=8.0,
    seed=42,
    num_workers=0,
    device="auto",
    shuffle=True,
    run_name=None,
    checkpoint_root=None,
    save_path=None,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def specdiff_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def normalize_image_key(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = stem.replace("specular", "").replace("glossy", "").replace("diffuse", "")
    return re.sub(r"[^a-z0-9]+", "", stem)


def pair_image_paths(glossy_dir: Path, diffuse_dir: Path) -> list[tuple[Path, Path, str]]:
    glossy_candidates = [path for path in glossy_dir.iterdir() if path.suffix.lower() in VALID_EXTS]
    diffuse_candidates = [path for path in diffuse_dir.iterdir() if path.suffix.lower() in VALID_EXTS]

    glossy_files = {path.name: path for path in glossy_candidates}
    diffuse_files = {path.name: path for path in diffuse_candidates}
    exact_names = sorted(set(glossy_files) & set(diffuse_files))
    if exact_names:
        return [(glossy_files[name], diffuse_files[name], name) for name in exact_names]

    glossy_by_key: dict[str, Path] = {}
    for path in glossy_candidates:
        key = normalize_image_key(path.name)
        if key in glossy_by_key:
            raise RuntimeError(f"Duplicate glossy normalized key {key!r} in {glossy_dir}")
        glossy_by_key[key] = path

    diffuse_by_key: dict[str, Path] = {}
    for path in diffuse_candidates:
        key = normalize_image_key(path.name)
        if key in diffuse_by_key:
            raise RuntimeError(f"Duplicate diffuse normalized key {key!r} in {diffuse_dir}")
        diffuse_by_key[key] = path

    common_keys = sorted(set(glossy_by_key) & set(diffuse_by_key))
    if not common_keys:
        raise RuntimeError(f"No paired PSD samples found in {glossy_dir} and {diffuse_dir}")

    return [(glossy_by_key[key], diffuse_by_key[key], glossy_by_key[key].name) for key in common_keys]


def load_rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), resample=PIL_BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def validate_sample_pair(
    glossy_path: Path,
    diffuse_path: Path,
    image_size: int,
) -> str | None:
    try:
        glossy = load_rgb_tensor(glossy_path, image_size)
    except Exception as exc:
        return f"failed to load glossy image: {exc}"

    try:
        diffuse = load_rgb_tensor(diffuse_path, image_size)
    except Exception as exc:
        return f"failed to load diffuse image: {exc}"

    if not torch.isfinite(glossy).all():
        return "non-finite values in glossy image"
    if not torch.isfinite(diffuse).all():
        return "non-finite values in diffuse image"
    return None


def filter_valid_samples(
    samples: list[tuple],
    image_size: int,
    dataset_name: str,
) -> tuple[list[tuple], list[tuple[Path, Path, str]]]:
    valid_samples: list[tuple] = []
    skipped_samples: list[tuple[Path, Path, str]] = []

    for sample in samples:
        glossy_path = Path(sample[0])
        diffuse_path = Path(sample[1])
        error = validate_sample_pair(glossy_path, diffuse_path, image_size)
        if error is None:
            valid_samples.append(sample)
            continue

        skipped_samples.append((glossy_path, diffuse_path, error))
        print(f"[{dataset_name}] Skipping corrupted pair: {glossy_path} | {diffuse_path} | {error}")

    return valid_samples, skipped_samples


def build_trainer_batch_tensor(
    glossy: torch.Tensor,
    diffuse: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([glossy, diffuse], dim=0)


def resolve_split_dirs(psd_root: Path, split: str) -> tuple[Path, Path]:
    try:
        glossy_rel, diffuse_rel = SPLIT_DIRS[split]
    except KeyError as exc:
        raise ValueError(f"Unknown split: {split}") from exc
    return psd_root / glossy_rel, psd_root / diffuse_rel


def resolve_psd_root(psd_root: Path | None, split: str) -> Path:
    if psd_root is None:
        raise ValueError("Pass --psd-root with the PSD dataset root you want to use.")

    direct_root = Path(psd_root).expanduser().resolve()
    candidate_roots = [direct_root, direct_root / "PSD_Dataset"]

    for candidate in candidate_roots:
        glossy_dir, diffuse_dir = resolve_split_dirs(candidate, split)
        if glossy_dir.exists() and diffuse_dir.exists():
            return candidate

    raise FileNotFoundError(
        f"PSD dataset root {direct_root} does not contain the expected split directories for '{split}'."
    )


def is_local_object_dir(path: Path) -> bool:
    return path.is_dir() and (path / "glossy").exists() and (path / "diffuse").exists()


def has_local_paired_objects(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(is_local_object_dir(child) for child in path.iterdir())


def local_data_root(root: Path) -> Path:
    data_root = (root.parent / "data").resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Missing local data root: {data_root}")
    return data_root


def resolve_local_split_root(data_root: Path, split: str) -> Path:
    split_candidate = data_root / split
    if has_local_paired_objects(split_candidate):
        return split_candidate
    if has_local_paired_objects(data_root):
        return data_root
    raise FileNotFoundError(
        f"Could not find a paired local dataset split for '{split}' under {data_root}."
    )


class PSDTaskDataset(Dataset):
    def __init__(self, glossy_dir: Path, diffuse_dir: Path, image_size: int):
        self.glossy_dir = Path(glossy_dir)
        self.diffuse_dir = Path(diffuse_dir)
        self.image_size = int(image_size)

        if not self.glossy_dir.exists():
            raise FileNotFoundError(f"Missing glossy directory: {self.glossy_dir}")
        if not self.diffuse_dir.exists():
            raise FileNotFoundError(f"Missing diffuse directory: {self.diffuse_dir}")

        paired_samples = pair_image_paths(self.glossy_dir, self.diffuse_dir)
        self.samples, skipped_samples = filter_valid_samples(
            paired_samples,
            self.image_size,
            dataset_name="PSDTaskDataset",
        )
        self.skipped_samples = len(skipped_samples)
        if self.skipped_samples:
            print(f"[PSDTaskDataset] Retained {len(self.samples)} valid pairs and skipped {self.skipped_samples}.")
        if not self.samples:
            raise RuntimeError(f"No valid glossy/diffuse sample pairs found in {self.glossy_dir} and {self.diffuse_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        glossy_path, diffuse_path, name = self.samples[idx]
        glossy = load_rgb_tensor(glossy_path, self.image_size)
        diffuse = load_rgb_tensor(diffuse_path, self.image_size)
        return build_trainer_batch_tensor(glossy, diffuse)


class LocalPairedDataset(Dataset):
    def __init__(self, split_root: Path, image_size: int):
        self.split_root = Path(split_root)
        self.image_size = int(image_size)

        if not self.split_root.exists():
            raise FileNotFoundError(f"Missing local dataset directory: {self.split_root}")

        self.samples: list[tuple[Path, Path, str, str]] = []
        for object_dir in sorted(self.split_root.iterdir()):
            if not is_local_object_dir(object_dir):
                continue
            for glossy_path, diffuse_path, name in pair_image_paths(object_dir / "glossy", object_dir / "diffuse"):
                self.samples.append((glossy_path, diffuse_path, object_dir.name, name))

        if not self.samples:
            raise RuntimeError(f"No paired glossy/diffuse samples found in {self.split_root}")

        self.samples, skipped_samples = filter_valid_samples(
            self.samples,
            self.image_size,
            dataset_name="LocalPairedDataset",
        )
        self.skipped_samples = len(skipped_samples)
        if self.skipped_samples:
            print(f"[LocalPairedDataset] Retained {len(self.samples)} valid pairs and skipped {self.skipped_samples}.")

        if not self.samples:
            raise RuntimeError(f"No valid glossy/diffuse sample pairs found in {self.split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        glossy_path, diffuse_path, object_name, name = self.samples[idx]
        glossy = load_rgb_tensor(glossy_path, self.image_size)
        diffuse = load_rgb_tensor(diffuse_path, self.image_size)
        return build_trainer_batch_tensor(glossy, diffuse)


def build_data_loader(
    *,
    dataset_source: str,
    psd_root: Path | None,
    split: str,
    glossy_dir: str | None,
    diffuse_dir: str | None,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> tuple[DataLoader, dict[str, str]]:
    root = specdiff_root()
    if glossy_dir and diffuse_dir:
        glossy_path = Path(glossy_dir).expanduser().resolve()
        diffuse_path = Path(diffuse_dir).expanduser().resolve()
        dataset = PSDTaskDataset(glossy_path, diffuse_path, image_size=image_size)
        info = {
            "dataset_source": "explicit_dirs",
            "glossy_dir": str(glossy_path),
            "diffuse_dir": str(diffuse_path),
        }
    elif dataset_source == "local":
        local_root = local_data_root(root)
        split_root = resolve_local_split_root(local_root, split)
        dataset = LocalPairedDataset(split_root, image_size=image_size)
        info = {
            "dataset_source": "local",
            "data_root": str(local_root),
            "split_root": str(split_root),
        }
    else:
        discovered_root = resolve_psd_root(psd_root, split=split)
        glossy_path, diffuse_path = resolve_split_dirs(discovered_root, split)
        dataset = PSDTaskDataset(glossy_path, diffuse_path, image_size=image_size)
        info = {
            "dataset_source": "psd",
            "data_root": str(discovered_root),
            "glossy_dir": str(glossy_path),
            "diffuse_dir": str(diffuse_path),
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if hasattr(dataset, "skipped_samples"):
        info["skipped_samples"] = str(dataset.skipped_samples)
    return loader, info


def build_model(
    backbone_module: str,
    model_name: str,
    image_size: int,
    noise_steps: int,
    depth: int,
) -> torch.nn.Module:
    backbone_key = backbone_module.lower()
    model_key = model_name.lower()

    if backbone_key == "sdebackbone":
        if model_key in {"unetwithtransformer", "flex", "unet_transformer"}:
            return SDEBackbone.UNetWithTransformer(
                noise_steps=noise_steps,
                size=image_size,
                depth=depth,
                conditioning_channels=3,
            )
        if model_key in {"unetwithattention", "unet", "unet_attention"}:
            return SDEBackbone.UNetWithAttention(noise_steps=noise_steps, depth=depth)
        if model_key in {"uvit", "udit"}:
            return SDEBackbone.UViT(input_size=image_size, conditioning_channels=3)
        if model_key in {"unetwithuvit", "utflex"}:
            return SDEBackbone.UNetwithUViT(noise_steps=noise_steps, size=image_size, depth=depth)

    if backbone_key == "backbone":
        if model_key in {"unetwithtransformer", "flex", "unet_transformer"}:
            return Backbone.UNetWithTransformer(noise_steps=noise_steps, size=image_size, depth=depth)
        if model_key in {"unetwithattention", "unet", "unet_attention"}:
            return Backbone.UNetWithAttention(noise_steps=noise_steps, depth=depth)
        if model_key in {"uvit", "udit"}:
            return Backbone.UViT(input_size=image_size, conditioning_channels=3)
        if model_key in {"unetwithuvit", "utflex"}:
            return Backbone.UNetwithUViT(noise_steps=noise_steps, size=image_size, depth=depth)

    if backbone_key == "singbackbone":
        if model_key in {"unetwithtransformer", "flex", "unet_transformer"}:
            return SingBackbone.UNetWithTransformer(noise_steps=noise_steps, size=image_size, depth=depth)
        if model_key in {"unetwithattention", "unet", "unet_attention"}:
            return SingBackbone.UNetWithAttention(noise_steps=noise_steps, depth=depth)
        if model_key in {"uvit", "udit"}:
            return SingBackbone.UViT(input_size=image_size, conditioning_channels=3)
        if model_key in {"unetwithuvit", "utflex"}:
            return SingBackbone.UNetwithUViT(noise_steps=noise_steps, size=image_size, depth=depth)

    raise ValueError(f"Unsupported backbone/model combination: {backbone_module}/{model_name}")


def default_model_path(root: Path, model_name: str, run_name: str | None = None) -> Path:
    output_dir = root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = run_name if run_name else model_name
    return output_dir / f"{stem}.pth"


def format_value_for_name(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def Diffusion_Train(
    *,
    predictor: torch.nn.Module,
    loader: DataLoader,
    save_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    noise_steps: int = 200,
    epochs: int = 2000,
    lr: float = 1e-4,
    device: str | torch.device = "cuda",
    parameterization: str = "e",
    target_scale: float = 8.0,
    aifnetset: bool = True,
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if predictor is None:
        raise ValueError("predictor must be provided")
    if loader is None:
        raise ValueError("loader must be provided")

    root = specdiff_root()
    resolved_device = resolve_device(str(device))
    diffuser = diff.CosSchDiffuser(steps=noise_steps, device=resolved_device)

    checkpoint_dir = Path(checkpoint_root) if checkpoint_root else root / "checkpoints" / (run_name or predictor.__class__.__name__)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_save_path = checkpoint_dir / "checkpoint"

    trainer = T.Trainer(
        model=predictor,
        diffuser=diffuser,
        data_loader=loader,
        epochs=epochs,
        lr=lr,
        device=str(resolved_device),
        save_path=str(trainer_save_path),
        type=parameterization,
        target_scale=target_scale,
    )
    trained = trainer.train()
    if not trained:
        raise RuntimeError("Training did not complete successfully. Check logs/error_log.txt.")

    final_path = Path(save_path) if save_path else default_model_path(root, predictor.__class__.__name__, run_name=run_name)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": trained["last_model"].state_dict(),
        "ema_model_state_dict": trained["ema_model"].state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "noise_steps": noise_steps,
        "epochs": epochs,
        "lr": lr,
        "parameterization": parameterization,
        "target_scale": target_scale,
        "run_name": run_name,
    }
    if metadata:
        payload.update(metadata)

    torch.save(payload, final_path)
    print(f"Model saved to {final_path}")

    return {
        "last_model": trained["last_model"],
        "ema_model": trained["ema_model"],
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "final_model_path": final_path,
        "checkpoint_dir": checkpoint_dir,
    }


def default_run_name(args: argparse.Namespace) -> str:
    return (
        f"{args.dataset_source}_{args.backbone_module}_{args.model}_"
        f"img{args.image_size}_steps{args.noise_steps}_depth{args.depth}_"
        f"{args.parameterization}_ts{format_value_for_name(args.target_scale)}"
    )


def main() -> None:
    args = RUN_CONFIG
    set_seed(args.seed)

    root = specdiff_root()
    device = resolve_device(args.device)
    loader, data_info = build_data_loader(
        dataset_source=args.dataset_source,
        psd_root=Path(args.psd_root).expanduser().resolve() if args.psd_root else None,
        split=args.split,
        glossy_dir=args.glossy_dir,
        diffuse_dir=args.diffuse_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    model = build_model(
        backbone_module=args.backbone_module,
        model_name=args.model,
        image_size=args.image_size,
        noise_steps=args.noise_steps,
        depth=args.depth,
    )

    run_name = args.run_name or default_run_name(args)
    save_path = Path(args.save_path) if args.save_path else root / "models" / str(args.image_size) / f"{run_name}.pth"
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else root / "checkpoints" / run_name

    print(f"specdiff root: {root}")
    print(f"device: {device}")
    print(f"dataset source: {args.dataset_source}")
    print(f"backbone module: {args.backbone_module}")
    print(f"model: {args.model}")
    print(f"target scale: {args.target_scale}")
    for key, value in data_info.items():
        print(f"{key}: {value}")
    print(f"dataset size: {len(loader.dataset)}")
    print(f"batches per epoch: {len(loader)}")
    print(f"final model path: {save_path}")
    print(f"checkpoint root: {checkpoint_root}")

    Diffusion_Train(
        predictor=model,
        loader=loader,
        save_path=save_path,
        checkpoint_root=checkpoint_root,
        noise_steps=args.noise_steps,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        parameterization=args.parameterization,
        target_scale=args.target_scale,
        aifnetset=True,
        run_name=run_name,
        metadata={
            "dataset_source": args.dataset_source,
            "backbone_module": args.backbone_module,
            "model_name": args.model,
            "image_size": args.image_size,
            "depth": args.depth,
            "split": args.split,
            "target_scale": args.target_scale,
            **data_info,
        },
    )


if __name__ == "__main__":
    main()
