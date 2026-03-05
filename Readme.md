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
│  ├─ blender/                      # Blender renders
│  ├─ scenes/                # Mitsuba setup
│  ├─ train/                   # train files
│  ├─ test/                   # test files
│  └─ README.md                 # dataset specification and conventions
│
├─ pipeline/
│  └─ README.md                 # pipeline usage documentation
│
├─ models/
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