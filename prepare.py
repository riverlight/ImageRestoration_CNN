# -*- coding: utf-8 -*-

import argparse
import numpy as np
import h5py
import os
import cv2


def prepare(args):
    h5_file = h5py.File(args.output_path, 'w')
    noise_patchs = list()
    source_patchs = list()

    for count, image_file in enumerate(os.listdir(args.images_dir)):
        print(image_file)
        source_img = cv2.imread(os.path.join(args.images_dir, image_file))
        # source_img = cv2.resize(source_img, (256, 256))
        ret, noise_buf = cv2.imencode(".JPG", source_img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        noise_img = cv2.imdecode(noise_buf, 1)

        # source_img = np.array(source_img)
        # noise_img = np.array(noise_img)
        # noise_ds.append(noise_img)
        # source_ds.append(source_img)
        for i in range(0, source_img.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, source_img.shape[1] - args.patch_size + 1, args.stride):
                noise_patchs.append(source_img[i:i + args.patch_size, j:j + args.patch_size])
                source_patchs.append(noise_img[i:i + args.patch_size, j:j + args.patch_size])
        if count > 100 :
            break

    noise_ds = np.array(noise_patchs, dtype=np.uint8)
    source_ds = np.array(source_patchs, dtype=np.uint8)
    print(noise_ds.shape)
    h5_file.create_dataset('noise', data=noise_ds)
    h5_file.create_dataset('source', data=source_ds)
    h5_file.close()
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    args = parser.parse_args()

    prepare(args)
