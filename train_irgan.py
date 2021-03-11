# -*- coding: utf-8 -*-
"""
@说明     ：训练 IRGAN 模型
@作者     : Leon He
"""

import torch.backends.cudnn as cudnn
import torch as t
from torch import nn
from models import Generator, Discriminator
from IR_dataset import IRDataset
from utils import *
from torch.utils.data.dataloader import DataLoader


def eval(g, eval_dataloader, device, epoch):
    g.eval()
    epoch_psnr = AverageMeter()
    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with t.no_grad():
            preds = g(inputs).clamp(0.0, 1.0)
        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

    print(epoch, ', eval psnr: {:.2f}'.format(epoch_psnr.avg))
    del inputs, labels, preds


def main():
    train_file = "./datasets/train-src-q15-s48.h5"
    eval_file = './datasets/eval-src-q15-s48.h5'
    irresnet_checkpoint = './weights/best-resnet_305.pth'
    # 学习参数
    batch_size = 16
    start_epoch = 0
    epochs = 10000
    checkpoint_file = None       # irgan 的 checkpoint
    # checkpoint_file = "./weights/checkpoint_srgan_63.pth"
    workers = 8
    beta = 1e-3  # 判别损失乘子
    lr = 1e-4  # 学习率

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    generator = Generator()
    discriminator = Discriminator()
    optimizer_g = t.optim.Adam(params=generator.parameters(), lr=lr)
    optimizer_d = t.optim.Adam(params=discriminator.parameters(), lr=lr)
    train_dataset = IRDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers,
                                  pin_memory=False,
                                  drop_last=True)
    eval_dataset = IRDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # 损失函数
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # 加载 irresnet 预训练模型
    irresnetcheckpoint = t.load(irresnet_checkpoint)
    generator.net.load_state_dict(irresnetcheckpoint)

    if checkpoint_file is not None:
        checkpoint = t.load(checkpoint_file)
        start_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])

    for epoch in range(start_epoch, epochs+1):
        if epoch == int(epochs / 2):
            adjust_learning_rate(optimizer_g, 0.5)
            adjust_learning_rate(optimizer_d, 0.5)
        generator.train()
        discriminator.train()
        losses_c = AverageMeter()
        losses_a = AverageMeter()
        losses_d = AverageMeter()

        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = generator(inputs)
            content_loss = content_loss_criterion(outputs, labels)
            outputs_d = ((outputs - 0.5) *2).to(device) # 判别器输入范围：-1:1
            ir_discriminated = discriminator(outputs_d)
            adversarial_loss_g = adversarial_loss_criterion(
                ir_discriminated, torch.ones_like(ir_discriminated))
            perceptual_loss = content_loss + beta * adversarial_loss_g
            optimizer_g.zero_grad()
            perceptual_loss.backward()
            optimizer_g.step()
            losses_c.update(content_loss.item(), outputs.size(0))
            losses_a.update(perceptual_loss.item(), outputs.size(0))

            ## 判别器
            ir2_discriminated = discriminator(outputs_d.detach())
            labels_d = ((labels - 0.5) * 2).to(device)
            src_discriminated = discriminator(labels_d)
            adversarial_loss_d = adversarial_loss_criterion(ir2_discriminated, t.zeros_like(ir2_discriminated)) + \
                                adversarial_loss_criterion(src_discriminated, t.ones_like(src_discriminated))
            optimizer_d.zero_grad()
            adversarial_loss_d.backward()
            optimizer_d.step()
            losses_d.update(adversarial_loss_d.item(), outputs.size(0))

            print("epoch : {}, batch : {}, loss : {} {} {}".format(epoch, i, losses_c.avg, losses_a.avg, losses_d.avg))
            print("loss : ", perceptual_loss.item(), content_loss.item(), adversarial_loss_g.item(), adversarial_loss_d.item())

        del inputs, labels, outputs, ir_discriminated, ir2_discriminated, src_discriminated, labels_d, outputs_d  # 手工清除掉缓存
        print("epoch id : ", epoch)
        print("loss : ", losses_c.avg, losses_a.avg, losses_d.avg)
        eval(generator, eval_dataloader, device, epoch)
        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
        }, 'weights/checkpoint_srgan_{}.pth'.format(epoch))

    pass


if __name__ == "__main__":
    main()
