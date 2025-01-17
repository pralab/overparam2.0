import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optimizers
import math
# 72 mnist and 98 for cifar10


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self._train_step = 0
        self._val_step = 0
        self._test_step = 0

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def epoch_step(self):
        self._train_step = 0
        self._val_step = 0

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def train_step(self, data, device):
        raise NotImplementedError()

    def val_step(self, data, device):
        raise NotImplementedError()

    def test_step(self, data, device):
        raise NotImplementedError()

    def init_weights(self, prev_model):
        pass    

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
    
    
class ExpandableFcReLu_mnist(nn.Module):
    def __init__(self, expansion: int, out_classes=10):
        super(ExpandableFcReLu_mnist, self).__init__()
        channels = 1 
        inp_img_size = 784 
        self.fc1 = nn.Linear(channels*inp_img_size, expansion)
        self.fc2 = nn.Linear(expansion, 8 * expansion)
        self.fc3 = nn.Linear(8 * expansion, out_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class RandomFourier(CustomModel):
    def __init__(self, expansion: int, out_classes = 10):
        super(RandomFourier, self).__init__()
        dim_input = 2352        # for CIFAR10 only
        self.hidden = nn.Linear(dim_input, expansion, bias=False)
        # nn.init.normal_(self.hidden.weight, std=0.04)
        # self.hidden.weight.requires_grad = False  # fix weights
        self.out = nn.Linear(expansion, out_classes, bias=False)
        # self._optimizer = optimizers.SGD(
        #     self.parameters(), lr=0.001, momentum=0.95)
        # self._loss_function = nn.MSELoss()
        # self._one_hot_label = None
        # self._dim_output = out_classes

    def forward(self, x):
        x = torch.cos(self.hidden(x))  # exp(i * x) = cos(x) (+ i*sin(x))
        # x = nn.Softmax(self.out(x))
        x = self.out(x)
        return x
    

class ResNet(CustomModel):
    def __init__(self, expansion: int, out_classes = 10):
        super(ResNet, self).__init__()
        out_channel=math.ceil(2**(2+float(expansion)/2))
        self.resnet=Network(out_channel)


    def forward(self, x):
        x=self.resnet(x)
        return x




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        # preactivation only for residual path
        y = self.bn1(x)            
        y = F.relu(y, inplace=True)
        y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y




class Network(nn.Module):
    def __init__(self, base_channels):
        super(Network, self).__init__()

        input_shape = (1,3,28,28)
        n_classes = 10

        block=BasicBlock
        n_blocks_per_stage=2

        n_channels = [
            base_channels,
            base_channels * 2 * block.expansion,
            base_channels * 4 * block.expansion
            # base_channels * 4 * block.expansion
        ]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)

        self.stage1 = self._make_stage(
            n_channels[0],
            n_channels[0],
            n_blocks_per_stage,
            block,
            stride=1)
        self.stage2 = self._make_stage(
            n_channels[0],
            n_channels[1],
            n_blocks_per_stage,
            block,
            stride=2)
        self.stage3 = self._make_stage(
            n_channels[1],
            n_channels[2],
            n_blocks_per_stage,
            block,
            stride=2)
        # self.stage4 = self._make_stage(
        #     n_channels[2],
        #     n_channels[3],
        #     n_blocks_per_stage,
        #     block,
        #     stride=2)
        # self.bn = nn.BatchNorm2d(n_channels[3])
        self.bn = nn.BatchNorm2d(n_channels[2])     # EDIT

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(block_name,
                                 block(
                                     in_channels,
                                     out_channels,
                                     stride=stride))
            else:
                stage.add_module(block_name,
                                 block(
                                     out_channels,
                                     out_channels,
                                     stride=1))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # x=self.stage4(x)      # EDIT
        x = F.relu(
            self.bn(x),
            inplace=True)  # apply BN and ReLU before average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# class ExpandableFcReLu_mnist(nn.Module):
#     def __init__(self, expansion: int, out_classes=10):
#         super(ExpandableFcReLu_mnist, self).__init__()
#         channels = 1 
#         inp_img_size = 784 
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(channels*inp_img_size, 3 * expansion)
#         self.fc2 = nn.Linear(3 * expansion, expansion)
#         self.fc3 = nn.Linear(expansion, 2 * expansion)
#         self.fc4 = nn.Linear(2 * expansion, out_classes)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim = 1)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
    
# class ExpandableFcReLu_cifar10(nn.Module):
#     def __init__(self, expansion: int, out_classes=10):
#         super(ExpandableFcReLu_cifar10, self).__init__()
#         channels = 3
#         inp_img_size = 1024
#         self.fc1 = nn.Linear(channels*inp_img_size, expansion)
#         self.fc2 = nn.Linear(expansion, 5 * expansion)
#         self.fc3 = nn.Linear(5 * expansion, out_classes)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
        