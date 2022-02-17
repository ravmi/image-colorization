from train import Trainer
from colorize_data import ColorizeData
import argparse

parser = argparse.ArgumentParser(description='Image colorization script')
parser.add_argument(
        'path',
        type=str,
        help='path to directory with train/val dataset')

parser.add_argument(
        '--lr',
        type=float,
        default=1e-6,
        help='learning rate')

parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs')

parser.add_argument(
        '--gpu',
        action='store_true')

args = parser.parse_args()

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'

data = ColorizeData(args.path, device=device)
t = Trainer(data, lr=args.lr, device=device, epochs=args.epochs)
t.train()
