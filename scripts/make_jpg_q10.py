# -*- coding: utf-8 -*-

import cv2
import os

if __name__ == "__main__":
    os.chdir("D:/workroom/project/riverlight/ImageRestoration_CNN")
    jpg_file = "./sample/gotham.jpg"
    file_ext = '.' + jpg_file.split('.')[-1]
    img = cv2.imread(jpg_file)
    print(img.shape)
    cv2.imwrite(jpg_file.replace(file_ext, "-q10" + file_ext), img, [int(cv2.IMWRITE_JPEG_QUALITY), 15])
    print('done')
