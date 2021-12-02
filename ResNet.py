import torch.nn as nn

# function defining a convolution layer for the network with a fixed kernal size of 3x3
def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)

# class defining a residual block that consists of two convolutional layers
# input to the residual block needs to be downsampled before its addition to the output of the residual block when:
#   1.) the convolutional layer 1 has a stride greater than 1 resulting in downsampling of output 
#   2.) number of input channels != number of output channels 
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, downsampling = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(out_channels, out_channels, stride = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsampling = downsampling
    
    def forward(self, x):
        residual = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampling:
            residual = self.downsampling(x)
        out += residual
        out = self.relu(out)
        return out

# class defining the Residual CNN model
# the network consists of:
#   1.) convolutional layer that takes in 3 input channels and gives out 16 channels
#   2.) convolutional block 
class ResNet(nn.Module):

    # blocks will be a list that defines the number of sequential residual blocks in a particular layer
    # for example: [2, 2, 2]
    def __init__(self, block_type, blocks, classes = 3):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace = True)
        self.layer1 = self.make_layer(block_type, 16, blocks[0], stride = 1)
        self.layer2 = self.make_layer(block_type, 32, blocks[1], stride = 2)
        self.layer3 = self.make_layer(block_type, 64, blocks[2], stride = 2)
        
        # change this depending on the input images' size
        self.avg_pool = nn.AvgPool2d(50)
        
        self.fc = nn.Linear(64, classes)
    
    # this method creates the layers of residal blocks needed for the network 
    def make_layer(self, block_type, out_channels, blocks, stride = 1):
        downsampling = None

        # downsampling needed if:
        #   1.) the convolutional layer 1 has a stride other than 1 resulting in downsampling of output 
        #   2.) number of input channels != number of output channels 
        if (self.in_channels != out_channels) or (stride != 1):
            downsampling = nn.Sequential(conv3x3(self.in_channels, out_channels, stride = stride),
                                         nn.BatchNorm2d(out_channels))

        # generating residual blocks
        residual_blocks = []
        residual_blocks.append(block_type(self.in_channels, out_channels, stride = stride, downsampling = downsampling))
        self.in_channels = out_channels
        for i in range(1, blocks):
            residual_blocks.append(block_type(self.in_channels, out_channels))
        return nn.Sequential(*residual_blocks)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out