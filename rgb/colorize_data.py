from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
import PIL
import tqdm

class ColorizeData(Dataset):
    def __init__(self, path, extensions=['jpg'], limit=50, device='cpu'):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.data = list()
        self.grayscales = list()
        self.colors = list()
        filenames = os.listdir(path)
        #filenames = filenames[:limit]
        for img_name in tqdm.tqdm(filenames):
            if img_name[-4:] != '.jpg':
                print(f'Skipped {img_name}, not jpg')
                continue

            img = PIL.Image.open(os.path.join(path, img_name))
            if img.mode != 'RGB':
                print(f'Skipped {img_name}, incorrect mode: {img.mode}')
                continue

            grayscale = self.input_transform(img)
            color = self.target_transform(img)
            self.grayscales.append(self.input_transform(img))
            self.colors.append(self.target_transform(img))

            self.data.append([grayscale.to(device), color.to(device)])
        print(f'Loaded: {len(self.data)} images of files {len(filenames)} from {path}')


    def __len__(self) -> int:
        return len(self.grayscales)

#    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    def __getitem__(self, index: int):
        return self.data[index]
#        return [self.grayscales[index], self.colors[index]]
