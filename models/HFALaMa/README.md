# 

# A Two-Stage Method for Specular Highlight Detection and Removal in Medical Images

This repository contains code for the MICCAI 2025 paper **"A Two-Stage Method for Specular Highlight Detection and Removal in Medical Images"**. The primary goal is to eliminate specular reflections on tissues and instruments during surgical procedures.

> **Note:** If you use this code, please cite the upcoming paper once it is published and indexed.

## Dataset
 CVC dataset download link：https://pan.baidu.com/s/1d8TOgcwZGD7f9aOqfudIOw?pwd=hyc4 提取码: hyc4
 
We currently release only the public **CVC-Clinic Dataset**. This dataset does **not** include ground-truth specular masks. All specular annotations and preprocessing scripts in this repository are products of our work; please cite our paper when using them.

## Code Overview

The pipeline is divided into two main stages:

1. **Stage 1 – Specular Detection & Initial Removal**

   * Detects specular highlights and performs a coarse removal.
   * Achieves approximately 90% removal accuracy.
2. **Stage 2 – Inpainting Refinement**

   * Refines the Stage 1 output to produce more realistic tissue appearance.

Currently, only partial testing code is available. The full implementation and private datasets will be released after the paper is indexed.

## Environment Setup

Install required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Some dependencies may need manual installation via `pip`.)*

## Usage

### Stage 1: Specular Detection & Removal Only

```bash
python predict1.py \
  /root/miccai/best.pth \
  /root/miccai/test/13938_A.png \
  --device cuda:0
```

* Replace paths with your own model weight (`.pth`) and test image.
* This script outputs:

  * **mask**: binary specular mask
  * **nohighlight**: image with highlights removed
  * **mask-d** (optional): dilated mask if dilation is specified

### Stage 2: Full Two-Stage Pipeline
Download the pre-trained models and place them in the lama-main\big-lama\models directory.
“https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip”

```bash
python run.py \
  --indir  test \
  --stage1 result_stage1 \
  --stage2 result_stage2 \
  --device cuda:0 \
  --dilate 2 \
  --kernel 5
```

* The script will:

  1. Run Stage 1 on all images in `test/` and save outputs into subfolders under `result_stage1/`:

     * `mask/`, `nohighlight/`, `mask-d/`
  2. Feed the dilated mask and de-highlighted images into LAMA for Stage 2, saving final inpainted results to `result_stage2/`.

> **Tip:** Modify the hard‑coded paths in `run.py` if your directory structure differs.

## Future Work

* Release full code and private datasets after paper indexing.
* Provide additional scripts for training and evaluation.

---

**Please remember to cite:**

> *A Two-Stage Method for Specular Highlight Detection and Removal in Medical Images*, MICCAI 2025.
