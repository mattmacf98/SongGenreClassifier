import torch
from torch.utils import data
import numpy as np
import random

""" A PyTorch dataset to handle our lyric images.
    Setting train to true pulls from the data/train directory, which holds 80% of the data.
    Setting it to false pulls from test, which holds the other 20%.
    Indexing the dataset returns a label and a PyTorch Tensor with shape (C, H, W).
    We only have one channel, so an extra bogus dimension is added to our images to satisfy
    PyTorch. Images fed into a CNN must have a channel dimension at index 0.
    Setting crop to an integer results in randomly cropped images,
    leaving it as None returns the full images

    NEW: This dataset preloads all of the data.
    ---
    NOTE: The data is ordered, so be sure to create a DataLoader with a RANDOM sampler
"""

class LyricImages(data.Dataset):
    def __init__(self, dataset_dir, preload=True, train=True, crop=None):
        subdir = "train" if train else "test"
        self.path = dataset_dir.strip('/') + '/' + subdir + '/'
        self.crop = crop
        self.labels = np.load(self.path + 'labels.npy')
        self.length = len(self.labels)
        self.data = self.preload_data() if preload else None

    def preload_data(self):
        return [self.load_example(i) for i in range(self.length)]
            
    def load_example(self, index):
        return np.load(self.path + "img_" + str(index) + '.npy')

    def __len__(self):
        return self.length

    def random_crop(self, img):
        rows, _ = img.shape
        if rows == self.crop:
            return img
        elif rows < self.crop:
            # Right now this is a bad thing, but we could either
            # 1. Pad the image to reach the desired crop
            # 2. Reprocess the dataset to increase the minimum image size
            raise Exception("Crop size [("+str(self.crop)+
                            ", 300)] is smaller than image size ["+str(img.shape)+"]")
        start = random.randint(0, rows - self.crop)
        return img[start:start + self.crop]
        
    def to_tensor(self, img):
        return torch.from_numpy(np.expand_dims(img, axis=0))

    def __getitem__(self, index):
        image = self.data[index] if self.data else self.load_example(index)

        if self.crop:
            image = self.random_crop(image)

        return self.labels[index], self.to_tensor(image)
