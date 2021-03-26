# -*- coding: utf-8 -*-

import torch as t
from models_nir6 import *
import eval
import utils
import copy
import numpy as np

device = 'cpu'
# old_pth = "d:/nir6_test.pth"
old_pth = "./weights/nir6_best.pth"
total = 0
percent = 0.5
base_number = 1
pruned_bn_num = 2

def eval_model(model):
    pruned_pth = "./weights/pruned.pth"
    t.save(model, pruned_pth)
    eval.eval_image(pruned_pth)
    utils.test()


def pre_prune(model, thres):
    pruned = 0
    lst_type = list()
    lst_shape = list()
    lst_bn_id = list()
    lst_bn_mask = list()

    bn_count = 0
    layer_count = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            lst_type.append('BatchNorm2d')
            if bn_count < pruned_bn_num:
                lst_bn_id.append(layer_count)
                weight_copy = m.weight.data.clone()
                weight_copy.to(device)
                # 要保留的通道
                mask = weight_copy.abs().gt(thres).float()
                remain_channels = t.sum(mask)
                # 如果全部剪掉的话就提示应该调小剪枝程度了
                if remain_channels == 0:
                    print('\r\n!please turn down the prune_ratio!\r\n')
                    remain_channels = 1
                    mask[int(t.argmax(weight_copy))] = 1

                # ******************规整剪枝******************
                v = 0
                n = 1
                if remain_channels > base_number:
                    while v < remain_channels:
                        n += 1
                        v = base_number * n
                    if remain_channels - (v - base_number) < v - remain_channels:
                        remain_channels = v - base_number
                    else:
                        remain_channels = v
                    if remain_channels > m.weight.data.size()[0]:
                        remain_channels = m.weight.data.size()[0]
                    remain_channels = t.tensor(remain_channels)

                    y, j = t.sort(weight_copy.abs())
                    thre_1 = y[-remain_channels]
                    mask = weight_copy.abs().ge(thre_1).float()

                # 剪枝掉的通道数个数
                pruned = pruned + mask.shape[0] - t.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                lst_shape.append(int(remain_channels))
                lst_bn_mask.append(mask.clone())
                print("mask shape : ", mask.shape)
                print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                      format(k, mask.shape[0], int(t.sum(mask)), (mask.shape[0] - t.sum(mask)) / mask.shape[0]))
            else:
                weight_copy = m.weight.data.clone()
                weight_copy.to(device)
                mask = t.ones(weight_copy.shape[0])
                remain_channels = t.sum(mask)
                lst_shape.append(int(remain_channels))
                lst_bn_mask.append(mask.clone())
            layer_count += 1
            bn_count += 1
        if isinstance(m, nn.Conv2d):
            lst_type.append('Conv2d')
            lst_shape.append((m.weight.data.shape[0], m.weight.data.shape[1]))
            layer_count += 1

    # 调整 lst_shape
    for bn_id in lst_bn_id:
        bn_shape = lst_shape[bn_id]
        print(bn_id, bn_shape)
        lst_shape[bn_id-1] = (bn_shape, lst_shape[bn_id-1][1])
    # 这是我的模型特点，用到了 concat 结构
    lst_bn_next_layer_id = model.lst_bn_next_layer_id
    lst_bn_next_cat = model.lst_bn_next_cat
    for id, bn_next_layer in enumerate(lst_bn_next_layer_id):
        lst_shape[bn_next_layer] = (lst_shape[bn_next_layer][0], sum([lst_shape[bn_next_cat] for bn_next_cat in lst_bn_next_cat[id]]))


    print(len(lst_type), lst_type)
    print(len(lst_shape), lst_shape)
    pruned_ratio = float(pruned / total)
    print('\r\n!预剪枝完成!')
    print('total_pruned_ratio: ', pruned_ratio)
    eval_model(model)
    print("lst_bn_mask : ", lst_bn_mask)
    return lst_shape, lst_bn_mask

def do_prune(newmodel, model, lst_bn_mask):
    lst_bn_layer_id = model.lst_bn_layer_id
    lst_bn_next_layer_id = model.lst_bn_next_layer_id
    lst_bn_next_cat = model.lst_bn_next_cat
    lst_bn_conv2d_id = [id-1 for id in lst_bn_layer_id]

    dict_count_mask = dict()
    count = 0
    bn_count = -1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            bn_count += 1
            mask = lst_bn_mask[bn_count]
            dict_count_mask[count] = mask
            print("mask : ", mask)
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            print("idx : ", idx)
            m1.weight.data = m0.weight.data[idx].clone()
            m1.bias.data = m0.bias.data[idx].clone()
            m1.running_mean = m0.running_mean[idx].clone()
            m1.running_var = m0.running_var[idx].clone()
            count += 1
        if isinstance(m0, nn.Conv2d):
            print(count, "conv2d")
            if count in lst_bn_conv2d_id[0:-1]:
                mask = lst_bn_mask[bn_count+1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx, :, :, :].clone()
                m1.bias.data = m0.bias.data[idx].clone()
            elif count in lst_bn_next_layer_id:
                # 先找到索引
                for cat_id, value in enumerate(lst_bn_next_layer_id):
                    if count==value:
                        break
                print("cat_id : ", cat_id)
                lst_cat = lst_bn_next_cat[cat_id]
                lst_mask = list()
                for cat in lst_cat:
                    lst_mask.append(dict_count_mask[cat])
                print("lst_mask : ", lst_mask)
                mask = t.cat(lst_mask, 0)
                print("cat_mask : ", t.cat(lst_mask, 0))
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[ :,idx, :, :].clone()
                if m0.bias is not None:
                    m1.bias.data = m0.bias.data.clone()
                pass
            else:
                m1.weight.data = m0.weight.data.clone()
                if m0.bias is not None:
                    m1.bias.data = m0.bias.data.clone()
            count += 1
            # idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # print("idx0 : ", idx0)
            # print("m0.weight.data shape ", m0.weight.data.shape)
            # if idx0.size == 1:
            #     idx0 = np.resize(idx0, (1,))
            # m1.weight.data = m0.weight.data[:, idx0, :, :].clone()
            # m1.bias.data = m0.bias.data.clone()



def main():
    global total
    model = t.load(old_pth).to(device)
    print('old model : ', model)
    for count, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
            print(m.weight.data.shape)
        # for name, params in m.named_parameters():
        #     print(count, name)
    print(total)

    # 确定剪枝的全局阈值
    bn = t.zeros(total)
    index = 0
    i = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    # 按照权值大小排序
    y, j = t.sort(bn)
    thre_index = int(total * percent)
    if thre_index == total:
        thre_index = total - 1
    # 确定要剪枝的阈值
    thre_0 = y[thre_index].to(device)

    # ********************************预剪枝*********************************
    lst_shape, lst_bn_mask = pre_prune(model, thre_0)

    # ********************************剪枝*********************************
    newmodel = NewIRNet6(lst_shape)
    do_prune(newmodel, model, lst_bn_mask)
    eval_model(newmodel)



if __name__=="__main__":
    print("hi, this is a bn-layer prune program")
    main()
