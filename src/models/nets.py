from typing import Any, Optional, Union

from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMActorCriticNet(nn.Module):
    def __init__(
        self,
        act_vocab_size: int,
        encoder_class: str = None,
        hidden_dim: int = None,
        input_shape: list[int] = None,
        **kwargs
    ):
        super().__init__()

        self.encoder = eval(encoder_class)(input_shape, **kwargs)
        test_in = torch.zeros((1,) + tuple(input_shape)).long()
        test_out = self.encoder(test_in)
        in_dim = math.prod(test_out.shape[1:])
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(in_dim, hidden_dim)
        self.critic_linear = nn.Linear(hidden_dim, 1)
        self.actor_linear = nn.Linear(hidden_dim, act_vocab_size)

    @property
    def device(self):
        return self.actor_linear.weight.device

    def clear(self) -> None:
        self.hx, self.cx = None, None

    """
    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]
    """

    def reset(self, n: int) -> None:
        self.hx = torch.zeros(n, self.hidden_dim, device=self.device)
        self.cx = torch.zeros(n, self.hidden_dim, device=self.device)

    def forward(
        self, inputs: torch.LongTensor, mask_padding: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        x = inputs[mask_padding] if mask_padding is not None else inputs
        x = self.encoder(x)
        if mask_padding is None:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(
                x, (self.hx[mask_padding], self.cx[mask_padding])
            )
        logits_actions = rearrange(self.actor_linear(self.hx), "b a -> b 1 a")
        means_values = rearrange(self.critic_linear(self.hx), "b 1 -> b 1 1")
        return logits_actions, means_values


class AtariEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        x = x.mul(2).sub(1)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)

        return x


class MessengerEncoder(nn.Module):
    def __init__(self, input_shape: int, embed_dim: int = None):
        super().__init__()
        n_entities = 17
        n_channels = input_shape[0]
        resnet_cfg = {
            "kernel": [3, 3, 3],
            "stride": [1, 2, 2],
            "padding": [2, 1, 1],
            "in_channels": [embed_dim * n_channels, 64, 64],
            "hidden_channels": [64, 64, 64],
            "out_channels": [64, 64, 64],
        }
        assert embed_dim is not None
        self.embedder = nn.Embedding(n_entities, embed_dim)
        self.resnet = ResNetEncoder(resnet_cfg)

    def forward(
        self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        assert inputs.min() >= 0 and inputs.max() < 17
        inputs = inputs.long().permute(0, 2, 3, 1)
        x = self.embedder(inputs)
        x = x.view(*x.shape[:3], -1).permute(0, 3, 1, 2)
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        kernel_sizes = cfg["kernel"]
        strides = cfg["stride"]
        paddings = cfg["padding"]
        in_channels = cfg["in_channels"]
        out_channels = cfg["out_channels"]
        hidden_channels = cfg["hidden_channels"]

        encoder_layers = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=hidden_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                ),
                nn.ReLU(),
                BasicBlock(
                    inplanes=hidden_channels[i],
                    planes=out_channels[i],
                    padding=1,
                    norm_layer=MyGroupNorm,
                    downsample=nn.Conv2d(hidden_channels[i], out_channels[i], 1),
                ),
            )
            for i in range(len(in_channels))
        ]

        self.num_layers = len(encoder_layers)
        self.model = nn.Sequential(*encoder_layers)

    def forward(self, in_data):
        return self.model(in_data)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        padding=1,
        dilation=0,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MyGroupNorm(nn.Module):
    # num_channels: num_groups
    GROUP_NORM_LOOKUP = {
        16: 2,  # -> channels per group: 8
        32: 4,  # -> channels per group: 8
        64: 8,  # -> channels per group: 8
        128: 8,  # -> channels per group: 16
        256: 16,  # -> channels per group: 16
        320: 16,  # -> channels per group: 16
        512: 32,  # -> channels per group: 16
        640: 32,  # -> channels per group: 16
        1024: 32,  # -> channels per group: 32
        2048: 32,  # -> channels per group: 64
    }

    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(
            num_groups=self.GROUP_NORM_LOOKUP[num_channels], num_channels=num_channels
        )

    def forward(self, x):
        x = self.norm(x)
        return x


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)
