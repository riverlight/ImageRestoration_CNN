# -*- coding: utf-8 -*-

import argparse
import torch as t
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import cv2

from models import IRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

    model = IRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in t.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    image = cv2.imread(args.image_file).astype(np.float32)
    image = t.from_numpy(image).to(device) / 255
    image = image.permute(2, 0, 1).unsqueeze(0)

    with t.no_grad():
        preds = model(image).clamp(0.0, 1.0)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    cv2.imwrite(args.image_file.replace('.', '-ir.'), preds)

    print('done')
