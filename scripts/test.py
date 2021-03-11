# -*- encoding: utf-8 -*-

import torch as t
import cv2
from PIL import Image
import torchvision.transforms.functional as FT


imagenet_mean = t.FloatTensor([0.485, 0.456,
                                   0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = t.FloatTensor([0.229, 0.224,
                                  0.225]).unsqueeze(1).unsqueeze(2)

def test():
    image_name = 'd:/workroom/testroom/old1215/sr0min1.jpg'
    img = cv2.imread(image_name)
    img = img.transpose(2, 0, 1)
    img = t.from_numpy(img)
    print(img.shape)


    img = Image.open(image_name, mode='r')
    img = img.convert('RGB')
    img = img.crop((0, 0, 4, 4))

    print(type(img))
    img = FT.to_tensor(img)
    img = t.ones_like(img)
    print(img)
    print(img.shape)
    print(imagenet_mean.shape)
    print(imagenet_std.shape)
    img = (img-imagenet_mean)/imagenet_std
    print(img.shape)
    print(img)
    # print(img.shape)

    pass


if __name__ == "__main__":
    test()
