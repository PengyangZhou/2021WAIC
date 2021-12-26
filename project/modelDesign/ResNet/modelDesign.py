import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output, identity_data)
        return output


def make_layer(block, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block())
    return nn.Sequential(*layers)


class AIModel(nn.Module):
    def __init__(self, ):
        super(AIModel, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        self.residual = make_layer(ResidualBlock, 8)
        self.conv_mid = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        self.conv_output = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(3, 3), padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, snr):
        x = x.permute(0, 3, 1, 2)
        _, _, H, W = x.shape
        size = (2, 7)
        up_size = (H * size[0], W * size[1])
        out = nn.UpsamplingNearest2d(size=up_size)(x)
        out = self.conv_input(out)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out, residual)
        out = self.conv_output(out)
        out = out.permute(0, 2, 3, 1)
        return out
