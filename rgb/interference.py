from train import Trainer
from colorize_data import ColorizeData
import argparse
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

parser.add_argument(
        '--gpu',
        action='store_true')

args = parser.parse_args()

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'


model = torch.load(args.model_path, map_location='cpu')
img = PIL.Image.open(args.img_path).convert('RGB')

resize = T.Compose([T.ToTensor(),
                                  T.Resize(size=(256,256)),
                                  T.Grayscale(),
                                  T.Normalize((0.5), (0.5))
                                  ])
x = resize(img)

model.eval()
out = model(torch.unsqueeze(x, dim=0))
img_colorized = T.ToPILImage()(out[0] * 0.5 + 0.5)
img_colorized.save(args.output_path)
