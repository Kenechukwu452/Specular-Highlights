###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright (c) 2017, Soumith Chintala. All rights reserved.
###############################################################################
"""
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
Introduced partial convolutions based padding for convolutional layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from prepo.models.partialconv2d import PartialConv2d

__all__ = [
    "PDResNet",
    "pdresnet18",
    "pdresnet34",
    "pdresnet50",
    "pdresnet101",
    "pdresnet152",
]


model_urls = {
    "pdresnet18": "",
    "pdresnet34": "",
    "pdresnet50": "",
    "pdresnet101": "",
    "pdresnet152": "",
}


def _prepare_mask(x, mask):
    if mask is None:
        return None
    if mask.ndim != 4:
        raise ValueError(f"Expected mask shape [B,1,H,W], got {tuple(mask.shape)}")
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)
    if mask.shape[-2:] != x.shape[-2:]:
        mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
    return mask.to(device=x.device, dtype=x.dtype)


def _merge_masks(mask_a, mask_b, size):
    if mask_a is None:
        mask = mask_b
    elif mask_b is None:
        mask = mask_a
    else:
        if mask_a.shape[-2:] != size:
            mask_a = F.interpolate(mask_a, size=size, mode="nearest")
        if mask_b.shape[-2:] != size:
            mask_b = F.interpolate(mask_b, size=size, mode="nearest")
        mask = torch.maximum(mask_a, mask_b)
    if mask is None:
        return None
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="nearest")
    return mask


def _pool_mask(mask, *, kernel_size, stride, padding):
    if mask is None:
        return None
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=stride, padding=padding)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with partial-conv padding."""
    return PartialConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        return_mask=True,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return PartialConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        return_mask=True,
    )


class PartialDownsample(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super().__init__()
        self.conv = conv1x1(inplanes, outplanes, stride=stride)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x, mask=None):
        mask = _prepare_mask(x, mask)
        x, mask = self.conv(x, mask)
        x = self.bn(x)
        return x, mask


class PartialStage(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, mask=None):
        for block in self.blocks:
            x, mask = block(x, mask)
        return x, mask


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask=None):
        mask = _prepare_mask(x, mask)
        residual = x
        residual_mask = mask

        out, out_mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, out_mask = self.conv2(out, out_mask)
        out = self.bn2(out)

        if self.downsample is not None:
            residual, residual_mask = self.downsample(x, mask)

        out += residual
        out = self.relu(out)
        return out, _merge_masks(out_mask, residual_mask, out.shape[-2:])


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask=None):
        mask = _prepare_mask(x, mask)
        residual = x
        residual_mask = mask

        out, out_mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, out_mask = self.conv2(out, out_mask)
        out = self.bn2(out)
        out = self.relu(out)

        out, out_mask = self.conv3(out, out_mask)
        out = self.bn3(out)

        if self.downsample is not None:
            residual, residual_mask = self.downsample(x, mask)

        out += residual
        out = self.relu(out)
        return out, _merge_masks(out_mask, residual_mask, out.shape[-2:])


class PDResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(PDResNet, self).__init__()
        self.conv1 = PartialConv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            return_mask=True,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = PartialDownsample(
                self.inplanes,
                planes * block.expansion,
                stride=stride,
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return PartialStage(layers)

    def forward_encoder(self, x, mask=None):
        mask = _prepare_mask(x, mask)

        stem, stem_mask = self.conv1(x, mask)
        stem = self.bn1(stem)
        stem = self.relu(stem)

        x = self.maxpool(stem)
        pooled_mask = _pool_mask(stem_mask, kernel_size=3, stride=2, padding=1)

        enc1, mask1 = self.layer1(x, pooled_mask)
        enc2, mask2 = self.layer2(enc1, mask1)
        enc3, mask3 = self.layer3(enc2, mask2)
        enc4, mask4 = self.layer4(enc3, mask3)

        return (stem, enc1, enc2, enc3, enc4), (stem_mask, mask1, mask2, mask3, mask4)

    def forward(self, x, mask=None):
        (_, _, _, _, x), _ = self.forward_encoder(x, mask)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def pdresnet18(pretrained=False, **kwargs):
    """Construct a PDResNet-18 model."""
    model = PDResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["pdresnet18"]))
    return model


def pdresnet34(pretrained=False, **kwargs):
    """Construct a PDResNet-34 model."""
    model = PDResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["pdresnet34"]))
    return model


def pdresnet50(pretrained=False, **kwargs):
    """Construct a PDResNet-50 model."""
    model = PDResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["pdresnet50"]))
    return model


def pdresnet101(pretrained=False, **kwargs):
    """Construct a PDResNet-101 model."""
    model = PDResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["pdresnet101"]))
    return model


def pdresnet152(pretrained=False, **kwargs):
    """Construct a PDResNet-152 model."""
    model = PDResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["pdresnet152"]))
    return model
