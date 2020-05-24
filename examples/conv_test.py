from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

mnist_data = MNIST("data", download=True, transform=transform)
# TODO try and modify num_workers in DataLoader
data_loader = DataLoader(mnist_data,
                         batch_size=32,
                         shuffle=True)

num_classes = 10


class MNISTnet(nn.Module):
    def __init__(self):
        super(MNISTnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.fc1 = nn.Linear(12*5*5, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = F.relu(x)
        x = F.relu(x)
        return x


tensor = torch.rand((28, 28, 3))
net = MNISTnet()
out = net(tensor)
