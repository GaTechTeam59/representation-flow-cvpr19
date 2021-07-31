"""
This files includes a resent18 model for applying 3d convolutions over videos for classifications.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#### My own code implementations

def get_inplanes():
    return [64, 128, 256, 512]

def conv_3d_3s_kerenl(in_planes, out_planes, stride):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv_3d_3s_kerenl(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv_3d_3s_kerenl(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm3d(out_planes)

        self.conv2 = conv_3d_3s_kerenl(out_planes, out_planes, stride = 1)
        self.bn2 = nn.BatchNorm3d(out_planes)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                conv_3d_3s_kerenl(in_planes, out_planes, stride),
                nn.BatchNorm3d(out_planes))


    def forward(self, x):
        residual_shortcut = x

        print("basic block")
        print(residual_shortcut.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        print("block conv 1")
        print(x.shape)

        x = self.conv2(x)
        x  = self.bn2(x)

        print("block conv 2")
        print(x.shape)


        residual_shortcut = self.shortcut(residual_shortcut)

        print("match ")
        print(residual_shortcut.size())

        x += residual_shortcut
        output = self.relu(x)

        return output

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)

        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  out_planes=planes,
                  stride=stride,
                  downsample=None))

        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        print("lets see what comes into the resnet")
        print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        print("see what conv1 does")
        print(x.size())

        if not self.no_max_pool:
            x = self.maxpool(x)

        print("start the layers")

        x = self.layer1(x)

        print("done with layer 1")
        x = self.layer2(x)

        print("done with layer 2")
        x = self.layer3(x)
        print("done with layer 3")
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_resent18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)

    return model



