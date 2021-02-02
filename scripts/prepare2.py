# -*- coding: utf-8 -*-

import argparse
import numpy as np
import h5py
import os
import cv2

"""make dataset : source/noise/deblock """


def prepare(args):
    h5_file = h5py.File(args.output_path, 'w')
    noise_patchs = list()
    source_patchs = list()

    source_dir = os.path.join(args.dataset_dir, 'source')
    noise_dir = os.path.join(args.dataset_dir, args.type)
    lst_sources = os.listdir(source_dir)
    lst_noises = os.listdir(noise_dir)
    if len(lst_noises) != len(lst_sources):
        raise Exception("wrong...")
    for count, _ in enumerate(lst_sources):
        source_file = os.path.join(source_dir, lst_sources[count])
        noise_file = os.path.join(noise_dir, lst_noises[count])
        print(source_file, noise_file)
        source_img = cv2.imread(source_file).transpose(2, 0, 1)
        noise_img = cv2.imread(noise_file).transpose(2, 0, 1)
        for i in range(0, source_img.shape[1] - args.patch_size + 1, args.stride):
            for j in range(0, source_img.shape[2] - args.patch_size + 1, args.stride):
                noise_patchs.append(noise_img[:, i:i + args.patch_size, j:j + args.patch_size])
                source_patchs.append(source_img[:, i:i + args.patch_size, j:j + args.patch_size])

    noise_ds = np.array(noise_patchs, dtype=np.uint8)
    source_ds = np.array(source_patchs, dtype=np.uint8)
    print(noise_ds.shape)
    h5_file.create_dataset('noise', data=noise_ds)
    h5_file.create_dataset('source', data=source_ds)
    h5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='noise')
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=48)
    parser.add_argument('--stride', type=int, default=48)
    args = parser.parse_args()

    prepare(args)
