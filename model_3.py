import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    drop=0.01
    #INPUT BLOCK
    self.conv1= self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
        )
    #CONVOLUTIONAL BLOCK 1
    self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
        )
    self.maxpool1=nn.MaxPool2d(2,2)
    #TRANSITION BLOCK 1
    self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            
        )
    # CONVOLUTION BLOCK 2
    self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(drop)
        )

    # GAP - Adaptive Global Average Pooling
    self.gap = nn.AdaptiveAvgPool2d(1)

    self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(1, 1), padding=0, bias=False), # output_size = 1    RF: 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # output  RF: 28
        )

  def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.maxpool1(x)
      x = self.conv3(x)
      x = self.conv4(x)
      x = self.gap(x)
      x = self.conv5(x)
      x = x.view(-1, 10)
      return F.log_softmax(x,dim=-1)

def model_summary(model,input_size):
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))
    return model,input_size 