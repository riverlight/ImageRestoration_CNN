# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np


def main():
    work_dir = "D:/workroom/project/riverlight/ImageRestoration_CNN/sample/q10"
    lst_image = ['he-test0.jpg',
                 'he-test0-q10.jpg',
                 'he-test0-q10-ir.jpg']

    lst_image_final = list()
    width = 0
    height = 0
    image_res = None
    for i, jpg_file in enumerate(lst_image):
        jpg_file = os.path.join(work_dir, jpg_file)
        lst_image_final.append(jpg_file)
        img = cv2.imread(jpg_file)
        if i == 0:
            print(img.shape)
            width = img.shape[1]
            height = img.shape[0]
            image_res = np.copy(img)
            continue
        if height != img.shape[0]:
            img = cv2.resize(img, (int(height*img.shape[1]/img.shape[0]), height))
        print(image_res.shape, img.shape)
        image_res = np.hstack((image_res, img))

    cv2.imwrite(os.path.join(work_dir, ("res.jpg")), image_res)



if __name__ == "__main__":
    main()
