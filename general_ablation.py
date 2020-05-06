from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
# Cool package to get an insight of your model like you do in keras summary()
from torchsummary import summary
import inspect
from ablator import Ablator

# Parameters

batch_size = 64
test_batch_size = 64
epochs = 1
lr = 1.0
gamma = 0.7
no_cuda = True
seed = 1
log_interval = 10
save_model = True

use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)

sequential_model = nn.Sequential(nn.Conv2d(1, 3, 3, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(3, 4, 3, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(4, 5, 3, 1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2),
                                 nn.Dropout2d(0.25),
                                 nn.Flatten(),
                                 nn.Linear(9216, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(128, 10),
                                 nn.LogSoftmax(dim=1)
                                 )

# This one here is the model built with PyTorch functional
# model = Net().to(device)

# This one here is the sequential one
model = sequential_model

optimizer = optim.Adadelta(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

input_shape = (1, 28, 28)

ablator = Ablator(model, input_shape=input_shape)
ablated_model = ablator.ablate_layers([0, 1, 2], infer_activation=True)
print(ablated_model)
