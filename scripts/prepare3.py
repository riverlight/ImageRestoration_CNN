# -*- coding: utf-8 -*-

import os
import cv2
from IR_video import CDeblock


""" jpg-dir -> q10-dir -> deblock-dir """


def main():
    source_dir = "D:\\workroom\\tools\\dataset\\IR-dataset\\eval\\source"
    q15_dir = "D:\\workroom\\tools\\dataset\\IR-dataset\\eval\\q15"
    deblock_dir = "D:\\workroom\\tools\\dataset\\IR-dataset\\eval\\deblock"
    ir = CDeblock()

    for i, basename in enumerate(os.listdir(source_dir)):
        print(i, basename)
        jpg_file = os.path.join(source_dir, basename)
        q15_file = os.path.join(q15_dir, basename.replace('.jpg', '-q15.jpg'))
        deblock_file = os.path.join(deblock_dir, basename.replace('.jpg', '-deblock.jpg'))
        img_jpg = cv2.imread(jpg_file)
        # print(basename, img_jpg.shape)
        cv2.imwrite(q15_file, img_jpg, [int(cv2.IMWRITE_JPEG_QUALITY), 15])
        img_q15 = cv2.imread(q15_file)
        img_deblock = ir.query(img_q15)
        cv2.imwrite(deblock_file, img_deblock, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        del img_deblock

if __name__ == "__main__":
    main()
