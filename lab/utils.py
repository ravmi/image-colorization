import torch
import numpy as np
from skimage import color
from PIL import Image
import torchvision.transforms as T

def make_image(grayscale, colors):
    img = torch.cat(((grayscale + 0.5) * 100, colors * 100), dim=0).cpu().detach().numpy()
    img = np.moveaxis(img, 0, -1)
    img = color.lab2rgb(img)
    img = Image.fromarray(np.uint8(img*255))

    return img

def image_to_input(img):
    resize = T.Compose([T.ToTensor(),
        T.Resize(size=(256,256)),
        T.ToPILImage()
        ])
    img = resize(img)

    rgb = np.asarray(img) / 255
    lab = color.rgb2lab(rgb)
    # first dimension of lab is grayscale, second and third are responsible for colors

    lab_channels_first = np.moveaxis(lab, -1, 0)

    x = torch.from_numpy(lab_channels_first[0:1, :, :]) / 100 - 0.5
    y = torch.from_numpy(lab_channels_first[1:3, :, :]) / 100

    return x, y
