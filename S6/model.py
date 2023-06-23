# Previous example has 9 layers
# the modified version has 11 layers (7 convolutional layers (last one is 1X1 Conv layer), 3 Maxpooling layers and 1 GAP Layer)
# Batch normalization is added after every convolutional layer
# padding is added for 6 convolutional layers.
#used dropout with probability=0
# In forward block - X.View two blocks is added

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.network = nn.Sequential(
        #  layer 1
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(), #  feature map size = (28, 28)
        nn.BatchNorm2d(8),
        nn.Dropout(p=0),
        #  layer 2
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Dropout(p=0),

        nn.MaxPool2d(2), #  feature map size = (14, 14)
        #  layer 3
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(), #  feature map size = (14, 14)
        nn.BatchNorm2d(16),
        nn.Dropout(p=0),
        #  layer 4
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(p=0),
        nn.MaxPool2d(2), #  feature map size = (7, 7)
        #  layer 5
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(), #  feature map size = (7, 7)
        nn.BatchNorm2d(32),
        nn.Dropout(p=0),
        #  layer 6
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout(p=0),
        nn.MaxPool2d(2), #  feature map size = (3, 3)
        #  output layer
        nn.Conv2d(32, 10, 1),
        nn.AvgPool2d(3)
    )

  def forward(self, x):
    x = x.view(-1, 1, 28, 28)
    x = self.network(x)
    x = x.view(-1, 10)
     
    y = F.log_softmax(x,dim=-1)
    return y

def model_summary(model,input_size):
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))
    return model,input_size                            
     