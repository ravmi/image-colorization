from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
import PIL
import tqdm
from skimage import io, color
from utils import image_to_input


class ColorizeData(Dataset):
    def __init__(self, path, extensions=['jpg'], device='cpu', test_run=False):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.resize = T.Compose([T.ToTensor(),
            T.Resize(size=(256,256)),
            T.ToPILImage()
            ])

        self.data = list()
        filenames = os.listdir(path)

        if test_run:
            filenames = filenames[:50]

        for img_name in tqdm.tqdm(filenames):
            if img_name[-4:] != '.jpg':
                print(f'Skipped {img_name}, not jpg')
                continue

            img = PIL.Image.open(os.path.join(path, img_name))
            if img.mode != 'RGB':
                print(f'Skipped {img_name}, incorrect mode: {img.mode}')
                continue
            x, y = image_to_input(img)
            '''
            img = self.resize(img)

            rgb = np.asarray(img) / 255
            lab = color.rgb2lab(rgb)
            # first dimension of lab is grayscale, second and third are responsible for colors

            lab_channels_first = np.moveaxis(lab, -1, 0)

            x = torch.from_numpy(lab_channels_first[0:1, :, :]) / 100 - 0.5
            y = torch.from_numpy(lab_channels_first[1:3, :, :]) / 100
            '''

            self.data.append([x.to(device, dtype=torch.float), y.to(device, dtype=torch.float)])
        print(f'Loaded: {len(self.data)} images of files {len(filenames)} from {path}')

    def __len__(self) -> int:
        return len(self.data)

#    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    def __getitem__(self, index: int):
        return self.data[index]
