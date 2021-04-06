# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('../')
# from models import IRResNet
from models_nir6 import *
import cv2
import torch as t
import numpy as np
import argparse
import utils
import time


class CDeblock:
    def __init__(self, weights_file='./weights/best-resnet_305.pth'):
        os.chdir('../')
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        # self.ir = IRResNet(n_blocks=3).to(self.device)
        # state_dict = self.ir.state_dict()
        # for n, p in t.load(weights_file, map_location=lambda storage, loc: storage).items():
        #     if n in state_dict.keys():
        #         state_dict[n].copy_(p)
        #     else:
        #         raise KeyError(n)
        self.ir = NewIRNet6().to(self.device)
        self.ir = t.load("./weights/pruned.pth").to(self.device)
        self.ir.eval()

    def query(self, img):
        image = img.astype(np.float32)
        image = t.from_numpy(image).to(self.device) / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        with t.no_grad():
            preds = self.ir(image).clamp(0.0, 1.0)
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
        del image
        return preds.astype(np.uint8)

    def query_file(self, image_file):
        image = cv2.imread(image_file)
        return self.query(image)


def main():
    s_mp4 = "d:/workroom/testroom/48.mp4"
    cap = cv2.VideoCapture(s_mp4)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    out = cv2.VideoWriter(s_mp4.replace('.mp4', '-ir.avi'), fourcc, fps, (width, height))
    ir = CDeblock()

    count = 0
    starttime = time.time()
    while True:
        if count % 25 == 0:
            print('frame id ', count)
        ret, frame = cap.read()
        if ret is not True:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        print('ts :', ts)
        print("cost time : ", time.time()-starttime)
        pred_img = ir.query(frame)
        # psnr = utils.calc_psnr(t.from_numpy(frame.astype(np.float32))/255, t.from_numpy(pred_img.astype(np.float32))/255)
        # print('psnr : ', psnr)
        # cv2.imwrite('1.jpg', pred_img)
        # break
        # r0, g0, b0 = cv2.split(frame)
        # r1, g1, b1 = cv2.split(pred_img)
        # new_img = cv2.merge([r1, g0, b0])
        # out.write(new_img)
        out.write(pred_img)
        count += 1
        if ts > 30:
            break
    out.release()
    cap.release()


if __name__ == "__main__":
    print("Hi, this is video IR program!")
    main()
    print('done')
