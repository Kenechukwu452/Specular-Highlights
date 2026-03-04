#!/usr/bin/env python3
# coding: utf-8
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from HFANET import HFANet

def single_image_test(model_path: str, image_path: str, device: str, dilate_iters: int, dilate_kernel: int):
    model = HFANet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    img = Image.open(image_path)
    M_orig, S, D = model.predict(img)

    M_dil = M_orig
    if dilate_iters > 0:
        mask_np = np.array(M_orig)
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=dilate_iters)
        M_dil = Image.fromarray(mask_np)

    fig, axes = plt.subplots(1, 5, num='SpecularRemoval', tight_layout=True)
    titles = ['Original image', 'Specular mask', 'Dilated mask', 'Specular image', 'Removed image']
    images = [img, M_orig, M_dil, S, D]
    for ax, im, title in zip(axes, images, titles):
        cmap = plt.cm.gray if 'mask' in title.lower() else None
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test specular removal on a single image')
    parser.add_argument('model', help='Path to the model weights (.pth)')
    parser.add_argument('image', help='Path to the input image')
    parser.add_argument('--device', default='cuda:0', help='Compute device, e.g., cpu or cuda:0')
    parser.add_argument('--dilate', type=int, default=0, help='Number of morphological dilation iterations to apply to the mask')
    parser.add_argument('--kernel', type=int, default=3, help='Morphological dilation kernel size (must be odd)')
    args = parser.parse_args()
    single_image_test(
        model_path    = args.model,
        image_path    = args.image,
        device        = args.device,
        dilate_iters  = args.dilate,
        dilate_kernel = args.kernel
    )
