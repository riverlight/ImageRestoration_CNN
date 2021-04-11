# -*- coding: utf-8 -*-

import torch
import torchvision
import cv2
import numpy as np
import caffe2.python.onnx.backend as backend
import onnxruntime as ort
import onnx
import sys
sys.path.append('../')
from models import IRResNet, IRTestNet
from models_nir9 import *

# onnx -> mnn
"""
./MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz
"""


def export_testnet():
    device = 'cpu'
    export_onnx_file = "d:/irtestnet.onnx"  # 目的ONNX文件名
    model = IRResNet(n_blocks=3).to(device)
    input_shape = (3, 96, 96)
    x = torch.randn(1, *input_shape).to(device)  # 生成张量
    torch.onnx.export(model, x, export_onnx_file, verbose=False, do_constant_folding=False,  # 是否执行常量折叠优化
                      input_names=["input11"],  # 输入名
                      output_names=["output44"])
    print("trans done")

def save_model():
    weights_file = '../weights/best-resnet_305.pth'
    model = IRResNet(n_blocks=3).to('cuda')
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()
    torch.save(model, '../weights/model-0310.pth')
    return


def model_test():
    device = 'cuda'
    # weights_file = '../weights/model-0310.pth'
    weights_file = "../weights/nir10_best.pth"
    # weights_file = "../weights/pruned.pth"
    # weights_file = "d:/nir9_test.pth"
    image_file = 'd:/workroom/testroom/v360.png'
    model = torch.load(weights_file).to(device)
    model.eval()
    image = cv2.imread(image_file).astype(np.float32)
    image = torch.from_numpy(image).to(device) / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        preds = model(image).clamp(0.0, 1.0)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    cv2.imwrite(image_file.replace('.', '-ir.'), preds)

    print('load done')
    input_shape = (3, 1280, 360)
    x = torch.randn(1, *input_shape).to(device)  # 生成张量
    export_onnx_file = "d:/nir10_best.onnx"  # 目的ONNX文件名
    torch.onnx.export(model, x, export_onnx_file, verbose=True, do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input11"],	# 输入名
                    output_names=["output44"],
                    dynamic_axes={'input11': {0: 'batch',2:'batch',3:'batch'}, 'output44': {0: 'batch',2:'batch',3:'batch'}})
    print("trans done")


def check_onnx():
    onnx_file = "../weights/best-resnet_305.onnx"
    model = onnx.load(onnx_file)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))
    print('check done')


def test_bycaffe2():
    onnx_file = "../weights/best-resnet_305.onnx"
    model = onnx.load(onnx_file)
    rep = backend.prepare(model, device="CPU")

    image_file = 'd:/workroom/testroom/v0.png'
    image = cv2.imread(image_file).astype(np.float32)
    image = image / 255
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0)[0:,0:,0:32*10,0:]
    print(image.shape)
    # exit(0)
    outputs = rep.run(image)
    preds = np.clip(outputs[0], 0.0, 1.0)
    preds = (preds*255.0).squeeze(0).transpose(1, 2, 0)
    cv2.imwrite(image_file.replace('.', '-onnx-ir.'), preds)
    print('test by caffe2 done')

def test_byort():
    onnx_file = "../weights/best-resnet_305.onnx"

    jsonDict = {}
    jsonDict['inputs'] = []
    jsonDict['outputs'] = []
    inputs = {}
    ort_session = ort.InferenceSession(onnx_file)
    model = onnx.load(onnx_file)
    for inputVar in ort_session.get_inputs():
        inp = {}
        inp['name'] = inputVar.name
        inp['shape'] = inputVar.shape
        inputs[inputVar.name] = np.random.uniform(0.1, 1.2, inputVar.shape).astype(np.float32)
        jsonDict['inputs'].append(inp)
    print([output.name for output in model.graph.output])
    for output in model.graph.output:
        jsonDict['outputs'].append(output.name)


if __name__=="__main__":
    # main()
    model_test()
    # check_onnx()
    # test_bycaffe2()
    # test_byort()
    # export_testnet()

