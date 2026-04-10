#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

import Diffuser as diff
import Run_Training as base_rt
import Trainer_v2 as T2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import ResidualLoader as residual_loader


# Edit these settings and run this file directly.
RUN_CONFIG = argparse.Namespace(
    residual_model_name="can",
    data_split="train",
    backbone_module="SDEBackbone",  # "SDEBackbone", "Backbone", "SingBackbone"
    model="unetwithtransformer",
    image_size=128,
    batch_size=4,
    noise_steps=200,
    epochs=200,
    lr=1e-4,
    depth=2,
    parameterization="e",
    target_scale=30.0,
    seed=42,
    num_workers=0,
    device="auto",
    shuffle=True,
    run_name=None,
    checkpoint_root=None,
    save_path=None,
    threshold_method="otsu",
    threshold=0.9,
    soft_gamma=1.0,
    kernel_size=3,
    fill_holes=True,
    residual_mode="subtractive",
)


def compute_object_selection_mask(
    glossy_img: torch.Tensor,
    diffuse_img: torch.Tensor,
) -> torch.Tensor:
    glossy_luma = glossy_img.mean(dim=0, keepdim=True)
    diffuse_luma = diffuse_img.mean(dim=0, keepdim=True)
    positive_residual = (glossy_luma - diffuse_luma).clamp_min(0.0)

    object_support = torch.maximum(glossy_luma, diffuse_luma)
    object_threshold = torch.maximum(
        torch.tensor(0.02, device=object_support.device, dtype=object_support.dtype),
        object_support.max() * 0.05,
    )
    object_mask = (object_support > object_threshold).float()
    if not torch.any(object_mask > 0.5):
        object_mask = (positive_residual > 0).float()
    return object_mask


class NotebookResidualBatchDataset(Dataset):
    def __init__(
        self,
        *,
        project_root: Path,
        model_name: str,
        split: str,
        image_size: int,
        threshold_method: str,
        threshold: float,
        soft_gamma: float,
        kernel_size: int,
        fill_holes: bool,
        residual_mode: str,
    ) -> None:
        self.project_root = Path(project_root)
        self.model_name = str(model_name)
        self.split = str(split)
        self.image_size = int(image_size)
        self.residual_mode = str(residual_mode)

        diffuse_dir = self.project_root / "data" / self.split / self.model_name / "diffuse"
        glossy_dir = self.project_root / "data" / self.split / self.model_name / "glossy"

        if not glossy_dir.exists():
            raise FileNotFoundError(f"Missing glossy directory: {glossy_dir}")
        if not diffuse_dir.exists():
            raise FileNotFoundError(f"Missing diffuse directory: {diffuse_dir}")

        self.glossy_dir = glossy_dir
        self.diffuse_dir = diffuse_dir
        self.base_dataset = residual_loader.PSDResidualDataset(
            diffuse_dir=str(diffuse_dir),
            glossy_dir=str(glossy_dir),
            resize_hw=self.image_size,
            threshold_method=threshold_method,
            threshold=threshold,
            kernel_size=kernel_size,
            soft_gamma=soft_gamma,
            fill_holes=fill_holes,
            residual_mode=residual_mode,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.base_dataset[idx]
        glossy = sample["input"]
        diffuse = sample["diffuse"]
        object_selection_mask = compute_object_selection_mask(glossy, diffuse)
        return torch.cat([glossy, diffuse, object_selection_mask], dim=0)


def build_notebook_residual_loader(
    *,
    project_root: Path,
    model_name: str,
    split: str,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    threshold_method: str,
    threshold: float,
    soft_gamma: float,
    kernel_size: int,
    fill_holes: bool,
    residual_mode: str,
) -> tuple[DataLoader, dict[str, str]]:
    dataset = NotebookResidualBatchDataset(
        project_root=project_root,
        model_name=model_name,
        split=split,
        image_size=image_size,
        threshold_method=threshold_method,
        threshold=threshold,
        soft_gamma=soft_gamma,
        kernel_size=kernel_size,
        fill_holes=fill_holes,
        residual_mode=residual_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    info = {
        "dataset_source": "runnotebook_residual_loader",
        "model_name": model_name,
        "split": split,
        "glossy_dir": str(dataset.glossy_dir),
        "diffuse_dir": str(dataset.diffuse_dir),
        "threshold_method": threshold_method,
        "threshold": str(threshold),
        "soft_gamma": str(soft_gamma),
        "kernel_size": str(kernel_size),
        "fill_holes": str(fill_holes),
        "residual_mode": residual_mode,
        "mask_kind": "object_selection",
        "batch_layout": "0:3 glossy | 3:6 diffuse | 6:7 object_selection_mask",
    }
    return loader, info


def Diffusion_Train_v2(
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
    residual_mode: str = "subtractive",
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if predictor is None:
        raise ValueError("predictor must be provided")
    if loader is None:
        raise ValueError("loader must be provided")

    root = base_rt.specdiff_root()
    resolved_device = base_rt.resolve_device(str(device))
    diffuser = diff.CosSchDiffuser(steps=noise_steps, device=resolved_device)

    checkpoint_dir = Path(checkpoint_root) if checkpoint_root else root / "checkpoints" / (run_name or predictor.__class__.__name__)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_save_path = checkpoint_dir / "checkpoint"

    trainer = T2.Trainer(
        model=predictor,
        diffuser=diffuser,
        data_loader=loader,
        epochs=epochs,
        lr=lr,
        device=str(resolved_device),
        save_path=str(trainer_save_path),
        type=parameterization,
        target_scale=target_scale,
        residual_mode=residual_mode,
    )
    trained = trainer.train()
    if not trained:
        raise RuntimeError("Training did not complete successfully. Check logs/error_log.txt.")

    final_path = Path(save_path) if save_path else base_rt.default_model_path(root, predictor.__class__.__name__, run_name=run_name)
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
        "residual_mode": residual_mode,
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
        f"rendered_{args.residual_model_name}_{args.backbone_module}_{args.model}_"
        f"img{args.image_size}_steps{args.noise_steps}_depth{args.depth}_"
        f"{args.parameterization}_ts{base_rt.format_value_for_name(args.target_scale)}_"
        f"{args.residual_mode}_{args.threshold_method}_mask"
    )


def main() -> None:
    args = RUN_CONFIG
    base_rt.set_seed(args.seed)

    root = base_rt.specdiff_root()
    device = base_rt.resolve_device(args.device)
    loader, data_info = build_notebook_residual_loader(
        project_root=PROJECT_ROOT,
        model_name=args.residual_model_name,
        split=args.data_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        threshold_method=args.threshold_method,
        threshold=args.threshold,
        soft_gamma=args.soft_gamma,
        kernel_size=args.kernel_size,
        fill_holes=args.fill_holes,
        residual_mode=args.residual_mode,
    )
    model = base_rt.build_model(
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
    print(f"project root: {PROJECT_ROOT}")
    print(f"device: {device}")
    print(f"residual model: {args.residual_model_name}")
    print(f"backbone module: {args.backbone_module}")
    print(f"model: {args.model}")
    print(f"parameterization: {args.parameterization}")
    print(f"target scale: {args.target_scale}")
    for key, value in data_info.items():
        print(f"{key}: {value}")
    print(f"dataset size: {len(loader.dataset)}")
    print(f"batches per epoch: {len(loader)}")
    print(f"final model path: {save_path}")
    print(f"checkpoint root: {checkpoint_root}")

    Diffusion_Train_v2(
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
        residual_mode=args.residual_mode,
        run_name=run_name,
        metadata={
            "dataset_source": "runnotebook_residual_loader",
            "residual_model_name": args.residual_model_name,
            "backbone_module": args.backbone_module,
            "model_name": args.model,
            "image_size": args.image_size,
            "depth": args.depth,
            "target_scale": args.target_scale,
            **data_info,
        },
    )


if __name__ == "__main__":
    main()
