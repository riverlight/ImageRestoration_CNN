# -*- coding: utf-8 -*-

from IR_dataset import IRDataset
from torch.utils.data.dataloader import DataLoader
import torch as t
from utils import AverageMeter, calc_psnr
from models_nir6 import *
from utils import AverageMeter, calc_psnr
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm


def main():
    train_file = "./datasets/denoise-train.h5"
    eval_file = "./datasets/denoise-eval.h5"
    outputs_dir = "./weights/"
    lr = 1e-4
    batch_size = 24
    num_epochs = 400
    num_workers = 8
    seed = 1018
    # best_weights = './weights/nir5_epoch_99.pth'
    best_weights = None
    start_epoch = 0
    # 稀疏训练
    s = 1e-5
    sr = True # 稀疏标志 sparsity-regularization

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    cudnn.benchmark = True
    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
    t.manual_seed(seed)
    if best_weights is not None:
        model = t.load(best_weights)
    else:
        model = NewIRNet6().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    train_dataset = IRDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  drop_last=True)
    eval_dataset = IRDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_epoch = 0
    best_psnr = 0.0
    for epoch in range(num_epochs-start_epoch):
        epoch += start_epoch
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as tq:
            tq.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))

            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                if sr:
                    updateBN(model, s)
                optimizer.step()

                tq.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                tq.update(len(inputs))

                # if i % 10 == 0:
                print(i, epoch_losses.avg)
        t.save(model, os.path.join(outputs_dir, 'nir6_epoch_{}.pth'.format(epoch)))
        model.eval()
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with t.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print(epoch, ', eval psnr: {:.2f}'.format(epoch_psnr.avg))
        del inputs, labels, preds
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            t.save(model, os.path.join(outputs_dir, 'nir6_best.pth'))

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))


if __name__=="__main__":
    main()
