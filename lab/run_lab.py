from train import Trainer
from colorize_data_lab import ColorizeData
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
        '--alpha',
        type=float,
        default=0.1,
        help='second loss multiplier (the higher the more intense colors)')

parser.add_argument(
        '--gpu',
        action='store_true')

parser.add_argument(
        '--loss_type',
        type=str,
        choices=['bright1', 'bright2'],
        default='bright1')

parser.add_argument(
        '--test_run',
        action='store_true')

args = parser.parse_args()

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'

print(args.test_run)
data = ColorizeData(args.path, device=device, test_run=args.test_run)
t = Trainer(data, lr=args.lr, device=device, epochs=args.epochs, alpha=args.alpha, loss_type=args.loss_type)
t.train()
