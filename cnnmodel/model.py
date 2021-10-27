import torch
import torch.nn as nn

# Specify number of classes
num_classes = 9
# Specify resolution of input
res = 16

# CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 1 input channel, 16 output channels
        self.conv_layer1 = self._conv_layer(1, 16)
        # 16 input channel, 64 output channels
        self.conv_layer2 = self._conv_layer(16, 64)
        # each channel is of dimension n*3, where n = (res-conv_kernel_size+1)/pool_stride_size
        self.fc1 = nn.Linear(self._get_conv_dim(self._get_conv_dim(res)) ** 3 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.drop=nn.Dropout(p=0.15) 
        self.batch=nn.BatchNorm1d(128)
        
    def _get_conv_dim(self, input_dim, kernel_size=3, stride=2):
        return (input_dim - kernel_size + 1) // stride
    
    def _conv_layer(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            # Conv3d with kernel size of 3*3*3
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=0),
            # LeakyReLU
            nn.LeakyReLU(),
            # MaxPool3D layer with kernel size 2*2*2, and stride size of 2 (by default)
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