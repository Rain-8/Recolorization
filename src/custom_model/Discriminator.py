import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3 + 18, 128, kernel_size=4, stride=2) #input to first layer has 21 channels, 3 for RGB rest 18 for palette embedding
        self.norm1 = nn.InstanceNorm2d(128)  # Updated to match conv1 output channels

        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.norm2 = nn.InstanceNorm2d(256)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2)
        self.norm3 = nn.InstanceNorm2d(512)

        self.conv4 = nn.Conv2d(512, 1, kernel_size=4, stride=2)
        self.norm4 = nn.InstanceNorm2d(512)

    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.conv1(x)))
        x = F.leaky_relu(self.norm2(self.conv2(x)))
        x = F.leaky_relu(self.norm3(self.conv3(x)))
        x = F.leaky_relu(self.norm4(self.conv4(x)))
        return torch.sigmoid(x)


#weights_init function is applied to the model to ensure that all convolutional layers start with random weights drawn from the same distribution, providing consistent and stable starting points across training runs
#Any normalization layers are initialized to have neutral weight and bias values.
#Any linear layers (if present) are initialized with Xavier normalization.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Norm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
