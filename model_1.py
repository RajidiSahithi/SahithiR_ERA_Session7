import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    #INPUT BLOCK
    self.conv1= self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )
    #CONVOLUTIONAL BLOCK 1
    self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )
    self.maxpool1=nn.MaxPool2d(2,2)
    #TRANSITION BLOCK 1 using kernal size 1 X 1 (Antman)
    self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
    # CONVOLUTION BLOCK 2
    self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        )
    #self.maxpool2=nn.MaxPool2d(2,2)

    # CONVOLUTIONAL BLOCK
    self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(6, 6), padding=0, bias=False),

        )

  def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.maxpool1(x)
      x = self.conv3(x)
      x = self.conv4(x)
      #x = self.maxpool2(x)
      x = self.conv5(x)
      x = x.view(-1, 16)
       
      y = F.log_softmax(x,dim=-1)
      return y

def model_summary(model,input_size):
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))
    return model,input_size 