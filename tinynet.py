import torch
from torch import nn
import torch.nn.functional as F
# 72 mnist and 98 for cifar10



class RandomReLU(nn.Module):
    def __init__(self, fan_in, lower=0.125, upper=0.333):
        super(RandomReLU, self).__init__()
        self.lower = lower
        self.upper = upper

        # Initialize negative slope parameter within a certain range
        self.negative_slope = nn.Parameter(torch.Tensor(1, fan_in, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Randomly initialize negative slope parameter within the specified range
        nn.init.uniform_(self.negative_slope, self.lower, self.upper)

    def forward(self, x):
        return F.relu(x) - self.negative_slope * F.relu(-x)
    

class ExpandableRandomReLU_cifar10(nn.Module):
    def __init__(self, expansion: int, out_classes=10):
        super(ExpandableRandomReLU_cifar10).__init__()
        channels = 3
        fc_inp = 98
        self.conv1 = nn.Conv2d(channels, expansion, 3, 1)
        self.fc1 = nn.Linear(expansion, 2 * expansion, 3, 2)
        self.fc2 = nn.Linear(fc_inp * expansion, out_classes)
        self.random_relu = RandomReLU(32)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.random_relu(x)
        x = self.fc2(x)
        return x

class ExpandableCNN_mnist(nn.Module):
    def __init__(self, expansion: int, out_classes=10):
        super(ExpandableCNN_mnist, self).__init__()
        fc_inp = 72
        channels = 1
        self.conv1 = nn.Conv2d(channels, expansion, 3, 1)
        self.conv2 = nn.Conv2d(expansion, 2 * expansion, 3, 2)
        self.fc1 = nn.Linear(fc_inp * expansion, out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    

class ExpandableCNN_cifar10(nn.Module):
    def __init__(self, expansion: int, out_classes=10):
        super(ExpandableCNN_cifar10, self).__init__()
        fc_inp = 98
        channels = 3
        self.conv1 = nn.Conv2d(channels, expansion, 3, 1)
        self.conv2 = nn.Conv2d(expansion, 2 * expansion, 3, 2)
        self.fc1 = nn.Linear(fc_inp * expansion, out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
class ExpandableCNN2_cifar10(nn.Module):
    def __init__(self, expansion: int, out_classes=10):
        super(ExpandableCNN2_cifar10, self).__init__()
        fc_inp = 98
        channels = 3
        self.conv1 = nn.Conv2d(channels, expansion, 3, 1)
        self.fc1 = nn.Linear(fc_inp * expansion, out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
# class ExpandableFcReLu_mnist(nn.Module):
#     def __init__(self, expansion: int, out_classes=10):
#         super(ExpandableFcReLu_mnist, self).__init__()
#         channels = 1 
#         inp_img_size = 784 
#         self.fc1 = nn.Linear(channels*inp_img_size, expansion)
#         self.fc2 = nn.Linear(expansion, 8 * expansion)
#         self.fc3 = nn.Linear(8 * expansion, out_classes)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    

class ExpandableFcReLu_mnist(nn.Module):
    def __init__(self, expansion: int, out_classes=10):
        super(ExpandableFcReLu_mnist, self).__init__()
        channels = 1 
        inp_img_size = 784 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels*inp_img_size, 3 * expansion)
        self.fc2 = nn.Linear(3 * expansion, expansion)
        self.fc3 = nn.Linear(expansion, 2 * expansion)
        self.fc4 = nn.Linear(2 * expansion, out_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class ExpandableFcReLu_cifar10(nn.Module):
    def __init__(self, expansion: int, out_classes=10):
        super(ExpandableFcReLu_cifar10, self).__init__()
        channels = 3
        inp_img_size = 1024
        self.fc1 = nn.Linear(channels*inp_img_size, expansion)
        self.fc2 = nn.Linear(expansion, 5 * expansion)
        self.fc3 = nn.Linear(5 * expansion, out_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
        