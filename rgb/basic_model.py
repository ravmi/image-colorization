import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
  def __init__(self, input_size=256):
    super(Net, self).__init__()
    

    # ResNet - First layer accepts grayscale images, 
    # and we take only the first few layers of ResNet for this task
    resnet = models.resnet18(num_classes=350)
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
    ## Upsampling Network

    self.upsample = nn.Sequential(
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
      )


  def forward(self, input):
    midlevel_features = self.midlevel_resnet(input)
    output = self.upsample(midlevel_features)
    return output
