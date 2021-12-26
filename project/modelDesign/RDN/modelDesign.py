import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=(1, 1))

    def forward(self, x):
        return x + self.lff(self.layers(x))


class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.G0 = 16  # num_features
        self.G = 16  # growth rate
        self.D = 8  # num_blocks
        self.C = 8  # num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(2, 16, kernel_size=(3, 3), padding=1)
        self.sfe2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=(1, 1)),
            nn.Conv2d(self.G0, self.G0, kernel_size=(3, 3), padding=1)
        )

        self.output = nn.Conv2d(self.G0, 2, kernel_size=(3, 3), padding=1)

    def forward(self, x, snr):
        x = x.permute(0, 3, 1, 2)
        _, _, H, W = x.shape
        size = (2, 7)
        up_size = (H * size[0], W * size[1])
        x = nn.UpsamplingNearest2d(size=up_size)(x)

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)
        return x
