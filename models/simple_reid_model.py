import torch
import torch.nn as nn


class SimpleReID(nn.Module):
    def __init__(self, input_size):
        super(SimpleReID, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # 只需一个输出，判断是否是同一个人
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x