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
│  ├─ train/                 # train files none
│  ├─ test/                  # test files none
│  └─ README.md              
│
├─ pipeline/
│  └─ README.md                 # pipeline usage documentation
│
├─ models/
│  ├─ MaskGen/                  #maskgeneration
│  │   └─src/                                     
│  │      └─ backbone 
│  └─ README.md
│                # path management utilities
│
├─ .gitignore
├─ environment                  # ignored
└─ README.md.                   