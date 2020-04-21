## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 7, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch1 = nn.BatchNorm2d(num_features=32, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.batch2 = nn.BatchNorm2d(num_features=64, affine=False, track_running_stats=False)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.drop3 = nn.Dropout(p=0.2)
        self.drop5 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*11*11, 2048)
        self.fc2 = nn.Linear(2048,512)
        self.fc3 = nn.Linear(512, 2*68) # size of output
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = x.view(x.size()[0], -1) # Flatten before dense layer
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        x = F.relu(self.fc2(x))
        x = self.drop5(x)
        x = self.fc3(x)
        
        return x
