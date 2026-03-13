# Synthetic Specular Highlight Generation and Removal

This repository contains research code for two linked tasks:

1. Generating soft highlight masks from diffuse/specular supervision.
2. Training partial-convolution models to remove specular highlights.

The current workflow is notebook-driven. The main reusable code lives in `pipeline/` and `sh_models/`, with Mitsuba-based scene generation under `pipeline/`.

## What is in the repository

```text
.
├── Readme.md
├── trainmodel.ipynb          # training experiments for detector + remover
├── runnotebook.ipynb         # rendering, loading, testing, and visualisation
├── pipeline/
│   ├── buildscene.py         # STL/PLY preprocessing + Mitsuba scene generation
│   ├── dhl_generator.py      # dynamic highlight mask generation
│   ├── PSDloader.py          # paired diffuse/specular dataset loader
│   └── generatem3sample.py   # Mitsuba render loop for paired diffuse/glossy views
├── sh_models/
│   ├── highlight_detector.py # soft highlight detector
│   ├── pconvAE.py            # partial-conv autoencoder for highlight removal
│   ├── pconvUnet.py          # experimental partial-conv U-Net variants with attention
│   ├── pd_resnet.py          # local partial-conv ResNet classification backbone
│   ├── 3chanelunet.py        # older experimental U-Net variant
│   └── prepo/                # vendored PartialConv implementation
├── weights/
│   ├── best_detector_softmask.pth
│   └── best_highlight_removal_rgb.pth
├── data/
│   ├── objects/             # source meshes and generated Mitsuba scenes
│   ├── train/
│   └── test/
└── outputs/



## Notebooks

- `trainmodel.ipynb`
  Trains the highlight removal autoencoder and the detector. The notebook currently uses absolute dataset paths, so those paths need to be changed locally before running it.

- `runnotebook.ipynb`
  Used for mask-generation experiments, mesh-to-scene generation, paired rendering, checkpoint loading, and visualisation. The notebook now includes preview cells for all rendered object datasets in `data/train/`.

## Notes

- `weights/` already contains trained checkpoints referenced by the notebooks.
- `data/test/` contains a small set of `.exr` examples, but the paired PSD loader currently reads standard image files rather than EXR.
- `sh_models/prepo` is mostly reference code; the highlight-removal workflow primarily uses `highlight_detector.py`, `pconvAE.py`, `pconvUnet.py`, and the local rendering/data pipeline.
