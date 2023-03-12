import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Type


class Bottleneck(nn.Module):
    """Bottleneck block for FPN"""

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Initialize the Bottleneck block for FPN.

        Args:
        - in_planes (int): number of input channels
        - planes (int): number of output channels
        - stride (int): stride to use for the convolutional layers
        """
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass for the Bottleneck block.

        Args:
        - x (torch.Tensor): input tensor of shape (batch_size, in_planes, H, W)

        Returns:
        - output (torch.Tensor): output tensor of shape (batch_size, self.expansion * planes, H', W')
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RetinaFPN(nn.Module):
    """RetinaFPN network in PyTorch."""

    def __init__(
        self, block: nn.Module, num_blocks: List[int], fpn_layer: List[int] = [3, 4]
    ) -> None:
        """
        Initialize the RetinaFPN network.

        Args:
        - block (nn.Module): bottleneck block to use
        - num_blocks (List[int]): number of blocks for each layer
        - fpn_layer (List[int]): indices of the layers to output for feature pyramid
        """
        super(RetinaFPN, self).__init__()

        self.f = 32
        self.in_plane_start = 12
        self.in_planes = self.in_plane_start
        self.fpn_layer = fpn_layer

        self.conv1_1 = nn.Conv2d(
            1, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Bottom-up layers
        self.layer2 = self._make_layer(block, self.f, num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 2 * self.f, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 4 * self.f, num_blocks[2], stride=2)
        self.layer5 = self._make_layer(block, 8 * self.f, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(
            32 * self.f, 4 * self.f, kernel_size=3, stride=2, padding=1
        )
        self.conv7 = nn.Conv2d(
            4 * self.f, 4 * self.f, kernel_size=3, stride=2, padding=1
        )

        # Top layer
        self.toplayer = nn.Conv2d(
            32 * self.f, 4 * self.f, kernel_size=1, stride=1, padding=0
        )  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(
            4 * self.f, 4 * self.f, kernel_size=3, stride=1, padding=1
        )
        self.smooth2 = nn.Conv2d(
            4 * self.f, 4 * self.f, kernel_size=3, stride=1, padding=1
        )
        self.smooth3 = nn.Conv2d(
            4 * self.f, 4 * self.f, kernel_size=3, stride=1, padding=1
        )

        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            16 * self.f, 4 * self.f, kernel_size=1, stride=1, padding=0
        )
        self.latlayer2 = nn.Conv2d(
            8 * self.f, 4 * self.f, kernel_size=1, stride=1, padding=0
        )
        self.latlayer3 = nn.Conv2d(
            4 * self.f, 4 * self.f, kernel_size=1, stride=1, padding=0
        )

    def _make_layer(
        self, block: Type[nn.Module], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a layer of the network.

        Args:
            block: The block to use in the layer.
            planes: The number of planes in the layer.
            num_blocks: The number of blocks to use in the layer.
            stride: The stride of the layer.

        Returns:
            A Sequential module representing the layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            layers.append(nn.BatchNorm2d(self.in_planes))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample and add two feature maps.

        Args:
            x: The top feature map to be upsampled.
            y: The lateral feature map.

        Returns:
            The added feature map.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass of the network.

        Args:
            x: The input to the network.

        Returns:
            A tuple containing the output of each FPN layer.
        """
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1_1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.toplayer(c5)

        output = []
        if 7 in self.fpn_layer:
            output.append(p7)
        if 6 in self.fpn_layer:
            output.append(p6)
        if 5 in self.fpn_layer:
            output.append(p5)
        if min(self.fpn_layer) <= 4:
            p4 = self._upsample_add(p5, self.latlayer1(c4))
        if 4 in self.fpn_layer:
            p4 = self.smooth1(p4)
            output.append(p4)
        if min(self.fpn_layer) <= 3:
            p3 = self._upsample_add(p4, self.latlayer2(c3))
        if 3 in self.fpn_layer:
            p3 = self.smooth2(p3)
            output.append(p3)
        if min(self.fpn_layer) <= 2:
            p2 = self._upsample_add(p3, self.latlayer3(c2))
        if 2 in self.fpn_layer:
            p2 = self.smooth3(p2)
            output.append(p2)
        output.reverse()

        return tuple(output)


def RetinaFPN50():
    return RetinaFPN(Bottleneck, [2, 4, 6, 3])


def RetinaFPN101():
    return RetinaFPN(Bottleneck, [2, 4, 23, 3])
