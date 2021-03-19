# -*- coding: utf-8 -*-

import os
import cv2


def main():
    src_dir = "D:\\workroom\\tools\\dataset\\voc2012\\JPEGImages"
    dst_dir = "./images"
    for count, imagefile in enumerate(os.listdir(src_dir)):
        print(imagefile)
        sfile = os.path.join(src_dir, imagefile)
        dfile = os.path.join(dst_dir, imagefile)
        img = cv2.imread(sfile)
        print(img.shape)
        img = img[100:196,100:196,]
        print(img.shape)
        cv2.imwrite(dfile, img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        print(count, imagefile)
        if count >= 1499:
            break
    return


if __name__=="__main__":
    main()
