# Synthetic Specular Highlight Generation and Removal

This repository contains research code for three linked tasks:

1. Generating soft highlight masks from diffuse/specular supervision.
2. Training models to remove specular highlights.
3. Training residual U-Nets that predict glossy-to-diffuse corrections.

The current workflow is notebook-driven. The main reusable code lives in `pipeline/` and `sh_models/`, with Mitsuba-based scene generation under `pipeline/`.

## What is in the repository

```text
.
├── Readme.md
├── trainmodel.ipynb              # detector + highlight-removal training
├── train_residual_unet.ipynb     # residual U-Net training experiments
├── masked_residual_restoration.ipynb
├── runnotebook.ipynb             # rendering, loading, testing, and visualisation
├── testnotebook.ipynb
├── pipeline/
│   ├── buildscene.py             # STL/PLY preprocessing + Mitsuba scene generation
│   ├── dhl_generator.py          # dynamic highlight mask generation
│   ├── PSDloader.py              # paired diffuse/glossy dataset loader
│   ├── ResidualLoader.py         # glossy-to-diffuse residual dataset loader
│   └── generatem3sample.py       # Mitsuba render loop for paired diffuse/glossy views
├── sh_models/
│   ├── highlight_detector.py     # soft highlight detector
│   ├── pconvAE.py                # partial-conv autoencoder for highlight removal
│   ├── pconvUnet.py              # partial-conv U-Net variants with attention
│   ├── pd_resnet.py              # local partial-conv ResNet classification backbone
│   ├── 3chanelunet.py            # U-Net used in the residual experiments
│   └── prepo/                    # vendored PartialConv implementation
├── weights/
│   ├── best_detector_softmask.pth
│   ├── best_highlight_removal_rgb.pth
│   └── residual U-Net checkpoints
├── data/
│   ├── objects/                 # source meshes and generated Mitsuba scenes
│   ├── train/
│   └── test/
└── outputs/
```

## Notebooks

- `trainmodel.ipynb`
  Trains the detector and the non-residual highlight-removal models (in-painting).

- `train_residual_unet.ipynb`
  Trains 3-channel and 4-channel residual U-Nets. The model predicts an RGB residual from the glossy input, then reconstructs the diffuse output with `glossy +/- residual`.

- `masked_residual_restoration.ipynb`
  Earlier masked-residual restoration experiments.

- `runnotebook.ipynb`
  Used for mask-generation experiments, mesh-to-scene generation, paired rendering, residual-loader inspection, checkpoint loading, and visualisation.

- `testnotebook.ipynb`
  Scratch notebook for local experiments.

## Notes

- The current workflow is notebook-first rather than package-first.
- `pipeline/PSDloader.py` loads paired glossy/diffuse images directly, and `pipeline/ResidualLoader.py` builds residual targets plus Otsu-based object masks.
- The residual notebook supports additive and subtractive residual conventions and saves focused-object and object-mask-only checkpoints under `weights/`.
- Most of the training code currently expects local absolute paths inside the notebooks.
- `sh_models/prepo/` is mostly reference code; the active experiments primarily use `highlight_detector.py`, `pconvAE.py`, `pconvUnet.py`, `pd_resnet.py`, and `3chanelunet.py`.
