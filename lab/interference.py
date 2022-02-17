from train import Trainer
from colorize_data_lab import ColorizeData
import argparse
from utils import make_image, image_to_input
import torch
import PIL

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
img_grayscale = PIL.Image.open(args.img_path).convert('RGB')A

input_transform = T.Compose([T.ToTensor(),
    T.Resize(size=(256,256)),
    T.Grayscale(),
    T.Normalize((0.5), (0.5))
    ])

data = input_transform(img_grayscale)

model.eval()
out = model(torch.unsqueeze(data, dim=0))
img_colorized = T.ToPILImage()(out * 0.5 + 0.5)
img_colorized.save(args.output_path)
