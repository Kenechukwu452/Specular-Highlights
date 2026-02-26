# Synthetic Specular Highlight Generation and Highlight Removal

This repository contains a modular research codebase for 

**1.** Generating **synthetic specular highlights** using physically based rendering and controlled lighting

**2.** Training and evaluating **highlight removal** models on paired synthetic data.

The project is organised so that **each model is self-contained** within its own folder, while the **data and data pipeline** live in a shared module accessible to all models. Visualisation notebooks and general helper utilities are kept at the repository root for convenience.

---

## Repository structure

```text
.
├─ data/
│  ├─ raw/                      # optional: original , meshes, HDRIs, source imagery
│  ├─ processed/                # dataset outputs (paired views, masks, and metadata)
│  ├─ splits/                   # train/val/test split files
│  └─ README.md                 # dataset specification and conventions
│
├─ pipeline/
│  ├─ render/                   # synthetic highlight generation (render configs, scripts)
│  ├─ transforms/               # augmentation, normalisation, masks, packaging
│  ├─ io/                       # loaders, writers, metadata schemas
│  ├─ configs/                  # dataset and render configuration templates
│  └─ README.md                 # pipeline usage documentation
│
├─ models/
│  ├─ model_a_name/
│  │  ├─ configs/
│  │  ├─ src/
│  │  ├─ scripts/               # train.py, eval.py, infer.py
│  │  ├─ checkpoints/           # optional, usually gitignored
│  │  └─ README.md              # model-specific instructions
│  │
│  ├─ model_b_name/
│  │  ├─ configs/
│  │  ├─ src/
│  │  ├─ scripts/
│  │  ├─ checkpoints/
│  │  └─ README.md
│  │
│  └─ ...
│
├─ notebooks/
│  ├─ 01_dataset_sanity_checks.ipynb
│  ├─ 02_render_preview.ipynb
│  ├─ 03_training_curves.ipynb
│  └─ 04_qualitative_results.ipynb
│
├─ helpers/
│  ├─ metrics.py                # shared metrics
│  ├─ viz.py                    # plotting helpers
│  ├─ misc.py                   # seeding, logging, convenience functions
│  └─ paths.py                  # path management utilities
│
├─ .gitignore
├─ environment                  # ignored
├─ requirements.txt             # pip requirements
├─ LICENSE                      # choose a licence
└─ README.md.                   # structure