import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PointNet, self).__init__()

        # MLP layers for transforming points
        self.mlp1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024)
        )

        # MLP layers for classification
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, points_num, input_size)

        # Apply MLP to each point
        x = self.mlp1(x)  # Shape: (batch_size, points_num, 1024)

        # Max pooling over points
        x = F.max_pool1d(x.transpose(1, 2), kernel_size=x.size(1)).squeeze()  # Shape: (batch_size, 1024)

        # Classification
        x = self.mlp2(x)  # Shape: (batch_size, num_classes)

        return x

# Example usage
# batch_size = 32
# points_num = 1024
# input_size = 3  # For XYZ coordinates
# num_classes = 10  # Example number of classes
#
# pointnet = PointNet(input_size, num_classes)
# input_data = torch.randn(batch_size, points_num, input_size)
# output = pointnet(input_data)
# print(output.shape)  # Should output: (batch_size, num_classes)