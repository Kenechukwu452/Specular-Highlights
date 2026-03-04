#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import sys
import tempfile
import shutil
import subprocess
import torch
import numpy as np
import cv2
from PIL import Image
from HFANET import HFANet

# Hard-coded paths
BEST_PTH = '/root/miccai/best.pth'
LAMA_DIR = '/root/miccai/lama-main'
LAMA_MODEL_DIR = '/root/miccai/lama-main/big-lama'

# Default directories
DEFAULT_INDIR = '/root/miccai/test'
DEFAULT_STAGE1 = '/root/miccai/result_stage1'
DEFAULT_STAGE2 = '/root/miccai/result_stage2'
STAGE1_SUBDIRS = ['mask', 'nohighlight', 'mask-d']


def remove_specular(hfa_path, img_path, device, dilate_iters, dilate_kernel):
    model = HFANet().to(device)
    state = torch.load(hfa_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    img = Image.open(img_path).convert('RGB')
    M_orig, _, D = model.predict(img)

    if dilate_iters > 0:
        m = np.array(M_orig)
        k = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        m_dil = cv2.dilate(m, k, iterations=dilate_iters)
        M_dil = Image.fromarray(m_dil)
    else:
        M_dil = M_orig

    return M_orig, M_dil, D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage1: HFANet removal + Stage2: LAMA inpainting')
    parser.add_argument('--indir',    default=DEFAULT_INDIR,  help='Input image directory')
    parser.add_argument('--stage1',   default=DEFAULT_STAGE1, help='Directory for HFANet results')
    parser.add_argument('--stage2',   default=DEFAULT_STAGE2, help='Directory for LAMA inpainting results')
    parser.add_argument('--device',   default='cuda:0',       help='cpu or cuda:0')
    parser.add_argument('--dilate',   type=int, default=2,     help='Mask dilation iterations')
    parser.add_argument('--kernel',   type=int, default=5,     help='Dilation kernel size (odd)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Prepare Stage1 directories
    for sub in STAGE1_SUBDIRS:
        os.makedirs(os.path.join(args.stage1, sub), exist_ok=True)
    # Ensure Stage2 directory exists
    os.makedirs(args.stage2, exist_ok=True)

    # Temporary folder for LAMA input
    tmp = tempfile.mkdtemp(prefix='lama_in_')

    try:
        # Stage1: HFANet spec removal + mask dilation
        for fn in sorted(os.listdir(args.indir)):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            inp = os.path.join(args.indir, fn)
            M_orig, M_dil, D = remove_specular(BEST_PTH, inp, device, args.dilate, args.kernel)
            base, ext = os.path.splitext(fn)

            # Save HFANet outputs
            M_orig.save(os.path.join(args.stage1, 'mask', fn))
            D.save(os.path.join(args.stage1, 'nohighlight', fn))
            M_dil.save(os.path.join(args.stage1, 'mask-d', fn))

            # Prepare for LAMA
            D.save(os.path.join(tmp, fn))
            M_dil.save(os.path.join(tmp, base + '_mask' + ext))

        # Stage2: LAMA inpainting with Hydra overrides
        env = os.environ.copy()
        env['PYTHONPATH'] = LAMA_DIR + os.pathsep + env.get('PYTHONPATH', '')

        predict_py = os.path.join(LAMA_DIR, 'bin', 'predict.py')
        cmd = [
            sys.executable,
            predict_py,
            f'hydra.run.dir={os.path.abspath(args.stage2)}',
            'hydra.job.chdir=false',
            f'model.path={LAMA_MODEL_DIR}',
            f'indir={tmp}',
            f'outdir={args.stage2}'
        ]

        subprocess.check_call(cmd, env=env)

    finally:
        shutil.rmtree(tmp)
