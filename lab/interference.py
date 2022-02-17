from train import Trainer
from colorize_data_lab import ColorizeData
import argparse
from utils import make_image, image_to_input
import torch
import PIL
from torchvision import transforms as T

parser = argparse.ArgumentParser(description='Image colorization script')
parser.add_argument(
        'img_path',
        type=str,
        help='path to grayscale jpg image')

parser.add_argument(
        'model_path',
        type=str,
        help='path to pretrained colorizing model')

parser.add_argument(
        'output_path',
        type=str,
        help='path where the colored image will be saved')

args = parser.parse_args()

model = torch.load(args.model_path, map_location='cpu')
img_grayscale = PIL.Image.open(args.img_path).convert('RGB')

x, y = image_to_input(img_grayscale)

x_batch = torch.unsqueeze(x, dim=0).float()
model.eval()
out = model(x_batch)
img_colorized = make_image(x_batch[0], out[0])
img_colorized.save(args.output_path)
