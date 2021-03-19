# -*- coding: utf-8 -*-

from IR_dataset import IRDataset
from torch.utils.data.dataloader import DataLoader
import torch as t
from utils import AverageMeter, calc_psnr
from models import *
import cv2
import numpy as np


def main():
    eval_file = "./weights/nir_best.pth"
    # eval_file = './weights/ir-0310.pth'
    # eval_file = "d:/tnet.pth"

    device = 'cuda'
    eval_dataset = IRDataset('datasets/train-src-q15-s48.h5')
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    net = t.load(eval_file)
    net = net.to(device)

    net.eval()
    epoch_psnr = AverageMeter()

    for count, data in enumerate(eval_dataloader):
        inputs, labels = data
        print(inputs.shape)
        inputs = inputs.to(device)
        labels = labels.to(device)
        with t.no_grad():
            preds = net(inputs).clamp(0.0, 1.0)
        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        # epoch_psnr.update(calc_psnr(inputs, labels), len(inputs))
        if count > 100:
            break
    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))


def eval_image():
    image_file = "sample/q10/he-test0-q10.jpg"
    out_file = image_file.replace('.', '-ir.')
    eval_file = "./weights/ir-0310.pth"
    # eval_file = "d:/tnet.pth"
    device = 'cuda'
    net = t.load(eval_file)
    net = net.to(device)
    net.eval()
    image = cv2.imread(image_file).astype(np.float32)
    image = t.from_numpy(image).to(device) / 255
    image = image.permute(2, 0, 1).unsqueeze(0)

    with t.no_grad():
        preds = net(image).clamp(0.0, 1.0)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    cv2.imwrite(out_file, preds)
    print('done : ', out_file)



if __name__=="__main__":
    # main()
    eval_image()

