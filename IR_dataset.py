# -*- coding: utf-8 -*-

import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2

class IRDataset(Dataset):
    def __init__(self, h5file):
        super(IRDataset, self).__init__()
        self.h5_file = h5file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['source'][idx].astype(np.float32)/255, f['noise'][idx].astype(np.float32)/255

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['source'])


def test():
    ds = IRDataset(".\\datasets\\1.h5")
    dl = DataLoader(dataset=ds, batch_size=1)
    for data in dl:
        source, noise = data
        source = source.numpy()*255
        noise = noise.numpy()*255
        cv2.imshow("a", noise[0, ...].astype(np.uint8))
        cv2.waitKey(0)
        # break

if __name__ == "__main__":
    test()
