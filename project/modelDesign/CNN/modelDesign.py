import torch
import torch.nn as NN
import torch.nn.functional as F


class AIModel(NN.Module):
    def __init__(self, ):
        self.num_layers = 9
        super(AIModel, self).__init__()
        for i in range(self.num_layers):
            if i == 0:
                in_channels = 2
                out_channels = 32
            elif i == 1:
                in_channels = 32
                out_channels = 64
            elif i == self.num_layers - 2:
                in_channels = 64
                out_channels = 32
            elif i == self.num_layers - 1:
                in_channels = 32
                out_channels = 2
            else:
                in_channels = 64
                out_channels = 64
            conv = NN.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
            torch.nn.init.normal_(conv.weight, 0, 0.05)
            torch.nn.init.constant_(conv.bias, 0)
            setattr(self, 'layer_%d' % (i + 1), conv)

    def forward(self, x, snr):
        x = x.permute(0, 3, 1, 2)
        _, _, H, W = x.shape
        size = (2, 7)
        up_size = (H * size[0], W * size[1])
        out = NN.UpsamplingNearest2d(size=up_size)(x)

        for i in range(self.num_layers):
            conv = getattr(self, 'layer_%d' % (i + 1))
            out = conv(out)
            out = F.leaky_relu(out, negative_slope=0.3)

        out = out.permute(0, 2, 3, 1)
        return out
