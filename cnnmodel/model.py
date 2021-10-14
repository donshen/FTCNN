import torch
import torch.nn as nn

# Specify number of classes
num_classes = 9

# CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer(1, 16)
        self.conv_layer2 = self._conv_layer(16, 64)
        self.fc1 = nn.Linear(2 ** 3 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.drop=nn.Dropout(p=0.15) 
        self.batch=nn.BatchNorm1d(128)
        
    def _conv_layer(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
   
    def forward(self, x):
        # print(x.size())
        x = self.conv_layer1(x)
        # print(x.size())
        x = self.conv_layer2(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        return x