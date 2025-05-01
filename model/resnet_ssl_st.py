import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialTransformer(nn.Module):
    def __init__(self, input_channels):
        super(SpatialTransformer, self).__init__()
        
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Fully connected layer for transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # Initialize the weights for identity transform
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

class ResNet18WithSTN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18WithSTN, self).__init__()
        
        # Load pre-trained ResNet18 layers
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace the first convolutional layer with spatial transformer followed by conv
        self.stn = SpatialTransformer(input_channels=3)
        
        # Adjust the fully connected layer for the desired number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Pass through spatial transformer
        x = self.stn(x)
        
        # Pass through ResNet layers
        x = self.resnet(x)
        
        return x

# Test the model with spatial transformer
if __name__ == "__main__":
    model = ResNet18WithSTN(num_classes=10)
    x = torch.randn(1, 3, 224, 224)  # Example input
    out = model(x)
    print("Output shape:", out.shape)
