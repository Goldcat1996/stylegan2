from io import BytesIO
import os
import lmdb
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from config import *

IMG_EXTENSIONS = (
                '.jpg', '.JPG', '.jpeg', '.JPEG',
                '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
            )


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        paths = path.split(',')
        if paths[-1] == '':
            paths = paths[:-1]
        if 'data.mdb' in os.listdir(paths[0]):
            path = paths[0]
            self.env = lmdb.open(
                path,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

            if not self.env:
                raise IOError('Cannot open lmdb dataset', path)

            with self.env.begin(write=False) as txn:
                self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

            self.resolution = resolution
            self.use_lmdb = True
        else:
            self.images = []
            for path in paths:
                self.images += make_dataset(path)
            self.length = len(self.images)
            self.use_lmdb = False

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.use_lmdb:
            with self.env.begin(write=False) as txn:
                key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
                img_bytes = txn.get(key)

            buffer = BytesIO(img_bytes)
        else:
            buffer = self.images[index]
        img = Image.open(buffer)
        img = np.array(img)
        if debug_real_color == 'RGB':
            img = img[..., ::-1]

        if img.dtype == 'uint16':
            img = cv2.convertScaleAbs(img, alpha=2 ** 8 / 2 ** 16)
        if img.shape[2] == 4:
            mask = img[..., 3:] / 255.0
            img = img[..., :3] * mask + 255.0 * (1 - mask)

        h, w = img.shape[:2]
        if h > w:
            img = np.pad(np.array(img), ((0, 0), (abs(h - w) // 2, abs(h - w) // 2), (0, 0)), 'edge')
        if h < w:
            img = np.pad(np.array(img), ((abs(h - w) // 2, abs(h - w) // 2), (0, 0), (0, 0)), 'edge')

        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)

        return img
