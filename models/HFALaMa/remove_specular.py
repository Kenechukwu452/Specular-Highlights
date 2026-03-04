#!/usr/bin/env python3
# coding: utf-8
import torch
import numpy as np
import cv2
from PIL import Image
from HFANET import HFANet  # 从 HFANET.py 中导入

def remove_specular(hfa_path, img_path, device, iters, kernel):
    model = HFANet().to(device)
    state = torch.load(hfa_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    M, _, D = model.predict(img)

    if iters > 0:
        m = np.array(M)
        k = np.ones((kernel, kernel), np.uint8)
        m = cv2.dilate(m, k, iterations=iters)
        M = Image.fromarray(m)

    return D, M
