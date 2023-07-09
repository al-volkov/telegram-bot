from typing import List
import numpy as np
import torch
from torch import nn


class GramMatrix(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = input.shape
        features = input.view(batch_size, channels, height * width)
        features_transposed = features.transpose(1, 2)
        gram_matrix = torch.bmm(features, features_transposed).div(channels * height * width)
        return gram_matrix


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 upsample: float = None) -> None:
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 4,
                 downsample: bool = False,
                 norm_layer=nn.BatchNorm2d) -> None:
        super().__init__()
        self.expansion = expansion
        self.downsample = downsample
        if self.downsample:
            self.residual_layer = nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride)
        self.conv_block = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            ConvLayer(out_channels, out_channels, kernel_size=3, stride=stride),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            return self.residual_layer(x) + self.conv_block(x)
        return x + self.conv_block(x)


class UpsampleResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, expansion: int = 4,
                 norm_layer=nn.BatchNorm2d) -> None:
        super().__init__()
        self.expansion = expansion
        self.residual_layer = UpsampleConvLayer(in_channels, out_channels * self.expansion,
                                                kernel_size=1, stride=1, upsample=stride)
        self.conv_block = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(out_channels, out_channels, kernel_size=3, stride=1, upsample=stride),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_layer(x) + self.conv_block(x)


class CoMatch(nn.Module):
    def __init__(self, shape: int, batch_size: int = 1) -> None:
        super(CoMatch, self).__init__()
        # batch_size is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.Tensor(1, shape, shape), requires_grad=True)
        # non-parameter buffer
        self.gram = torch.randn(batch_size, shape, shape, requires_grad=True)
        self.shape = shape
        self.product = torch.Tensor(batch_size, shape, shape)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.uniform_(0.0, 0.02)

    def set_target(self, target: torch.Tensor) -> None:
        self.gram = target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x is a 3D feature map
        self.product = torch.bmm(self.weight.expand_as(self.gram), self.gram)
        return torch.bmm(self.product.transpose(1, 2).expand(x.size(0), self.shape, self.shape),
                         x.view(x.size(0), x.size(1), -1)).view_as(x)

    def __repr__(self):
        return self.__class__.__name__ + f"(N x {self.shape})"


class MSGNet(nn.Module):
    # ngf - number of generator filter channels
    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 128, expansion: int = 4,
                 norm_layer=nn.InstanceNorm2d,
                 n_blocks: int = 6,
                 gpu_ids: List = None):
        super().__init__()
        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
        else:
            self.gpu_ids = []

        self.gram = GramMatrix()
        self.expansion = expansion
        self.norm_layer = norm_layer

        self.model1 = nn.Sequential(ConvLayer(input_nc, 64, kernel_size=7, stride=1),
                                    norm_layer(64),
                                    nn.ReLU(inplace=True),
                                    ResidualBlock(64, 32, 2, self.expansion, True, self.norm_layer),
                                    ResidualBlock(32 * self.expansion, ngf, 2, self.expansion, True, self.norm_layer))

        model = []
        self.co_match = CoMatch(ngf * self.expansion)
        model.append(self.model1)
        model.append(self.co_match)

        for i in range(n_blocks):
            model.append(ResidualBlock(ngf * self.expansion, ngf, 1, self.expansion, False, self.norm_layer))

        model += [UpsampleResidualBlock(ngf * self.expansion, 32, 2, self.expansion, self.norm_layer),
                  UpsampleResidualBlock(32 * self.expansion, 16, 2, self.expansion, self.norm_layer),
                  norm_layer(16 * self.expansion),
                  nn.ReLU(inplace=True),
                  ConvLayer(16 * self.expansion, output_nc, kernel_size=7, stride=1)]

        self.model = nn.Sequential(*model)

    def set_target(self, x):
        x1 = self.model1(x)
        xg = self.gram(x1)
        self.co_match.set_target(xg)

    def forward(self, x):
        return self.model(x)
