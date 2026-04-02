import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.module) :    
      def __init__(self) :
          super().__init__()
          self.conv1 = nn.Conv2d( 1 , 6 , 5 , padding = 0)
          self.conv2 = nn.Conv2d( 6 , 16 , 5 , padding = 0)
          self.conv3 = nn.Conv2d( 16 , 120 , 5 , padding = 0)
          self.fc1 = nn.Linear( 120   , 84)
          self.fc2 = nn.Linear( 84   , 10)

      def forward( self , x ) :
           x = F.max_pool2d(F.relu(self.conv1(x)),2)
           x = F.max_pool2d(F.relu(self.conv2(x)),2)
           x = F.relu(self.conv3(x))
           x = x.view(x.size(0) , -1 )
           x = F.relu(self.fc1(x))
           return self.fc2(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = pool
        
        if self.pool:
            self.pool_layer = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        if self.pool:
            x = self.pool_layer(x)
        
        return x


class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()       
        self.conv1 = ConvBlock(3, 64)                  
        self.conv2 = ConvBlock(64, 128, pool=True)     
      
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )
        
        self.conv3 = ConvBlock(128, 256, pool=True)    
        self.conv4 = ConvBlock(256, 512, pool=True)   
      
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )       
        self.pool = nn.AdaptiveAvgPool2d(1)            
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):       
        x = self.conv1(x)
        x = self.conv2(x)       
        x = x + self.res1(x) 
        x = self.conv3(x)
        x = self.conv4(x) 
        x = x + self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x