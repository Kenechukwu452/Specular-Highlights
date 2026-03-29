# FoilDiff

**FoilDiff** is a modular diffusion-based surrogate modeling framework for simulating 2D flow fields around airfoils. It extends classical Denoising Diffusion Implicit Models (DDIM) by enabling flexible denoising backbones tailored to the aerodynamic simulation domain. The framework supports five backbone architectures, combining convolutional inductive bias with transformer-based global reasoning for robust flow reconstruction across varying Reynolds numbers and angles of attack.

## Table of Contents

- [Overview](#overview)
- [Backbone Architectures](#backbone-architectures)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

FoilDiff predicts steady-state pressure and velocity fields around airfoils using denoising diffusion models trained on CFD-generated data. Unlike standard models fixed to one architecture, FoilDiff introduces a **backbone-agnostic approach** for the noise prediction network. This enables researchers to explore the trade-offs between convolutional and transformer-based architectures in a controlled generative setting.

Each diffusion model is conditioned on the airfoil geometry and flow parameters. Conditioning is applied spatially and temporally, enabling robust generalization to unseen physical regimes.

---

## Backbone Architectures

FoilDiff supports five backbone classes for the denoising network:

1. **UNet**  
   A conventional encoder–decoder with skip connections and local convolutions.

2. **Transformer (DiT-style)**  
   A patch-based transformer using global attention and sinusoidal time embedding. Suitable for large-scale flow patterns with spatial coherence.

3. **UNet with Transformer Mid-block (Flex)**  
   A hybrid model inspired by FLEX, inserting transformer blocks between encoder and decoder layers to fuse local and global representations.

4. **UViT (Unified Vision Transformer)**  
   A full transformer architecture using stacked self-attention layers across shallow and deep blocks with embedded conditioning.

5. **UNet with UViT Mid-block**  
   A hierarchical architecture combining the strong spatial priors of a UNet with the global capacity of a UViT at the bottleneck.

All backbones are modular and can be swapped by modifying the configuration or calling different constructors in `backbone.py`.

---

## Key Features

- **Backbone-agnostic diffusion training**
- **DDIM and DDPM inference modes**
- **Hierarchical time embedding**
- **Geometry and parameter conditioning**
- **Airfoil-aware preprocessing pipeline**
- **Support for hybrid transformer-convolutional models**

---

## Project Structure
    FoilDiff/
    ├── backbone.py             # Entry points for each denoising backbone
    ├── Diffuser.py             # Forward and reverse diffusion logic
    ├── Trainer.py              # Training script
    ├── Transformer.py          # Transformer modules and DiT / UViT implementations
    ├── process_opf_data.py     # Data preprocessing script
    ├── unifoil_pipeline.py     # Main execution pipeline
    ├── datasets/               # Dataset loading utilities
    ├── data/                   # Raw and processed simulation data
    ├── tests.py                # Backbone and sampler tests


## License

