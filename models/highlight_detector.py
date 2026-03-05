import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic conv block with BatchNoprm and ReLU
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_bn: bool = True, 
                 use_relu: bool = True):
        super().__init__()

        layers = [
                nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
                ]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class EncoderBlock(nn.Module):
    """
    Encoder block with two conv layers and optional downsampling
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool = True
                 ):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.downsample = nn.MaxPool2d(2, 2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = slef.conv1(x)
        x = self.conv2(x)
        return self.downsample(x), x # downsampled and skip connection

class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connections
    """
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int
                 ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch due to pooling
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Detector(nn.Module):
    """
    Highligh Detection Network (M_soft generator)
    It performs pixel-wise classification to detect highligh regions.
    It outputs M_soft, where:
        - Values close to 0 indicate strong highligh
        - Values close to 1 indicate non-highlight regions

    Args:
        in_channels: Number of input channels
        base_features: Number of features in the first encoder layer
        num_classes: Number of output classes (2 -> highlight/non-highlight)
    """
    def __init__(self,
                 in_channels: int = 1, # Grayscale
                 base_features: int = 64,
                 num_classes: int  = 2
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder path
        self.enc1 = EncoderBlock(in_channels, base_features)
        self.enc2 = EncoderBlock(base_features, base_features * 2)
        self.enc3 = EncoderBlock(base_features * 2, base_features * 4)
        self.enc4 = EncoderBlock(base_features * 4, base_features * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
                ConvBlock(base_features * 8, base_features * 16),
                ConvBlock(base_features * 16, base_features * 16)
                )

        # Decoder path
        self.dec1 = DecoderBlock(base_features * 16, base_features * 8, base_features * 8)
        self.dec2 = DecoderBlock(base_features * 8, base_features * 4, base_features * 4)
        self.dec3 = DecoderBlock(base_features * 4, base_features * 2, base_features * 2)
        self.dec4 = DecoderBlock(base_features * 2, base_features, base_features)

        # Classification layer
        self.out = nn.Conv2D(base_features, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """
        He initialisation
        """
        for m in self.module():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate M_soft
        Args:
            x: Input image tensor of shape (B, C, H, W)
        Returns:
            M_soft: Highlight detection map of shape (B, 1, H, W)
                Values in [0, 1] where 0 = strong highlight, 1 = non-highlight
        """
        # Encoder
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)

        # Bottleneck
        x = self.bottleneck(x4)

        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Classification
        logits = self.out(x)

        # Apply softmax to get probabilities
        # Channel 0: highlight probability
        # Channel 1: non-highlight probabilty
        probs = F.softmax(logits, dim=1)

        # M_soft is the non-highlight probability (channel 1)
        # So, 0 will be strong highlight, 1 --> non-highlight
        m_soft = probs[:, 1:2, :, :] # (B, 1, H, W)

        return m_soft
