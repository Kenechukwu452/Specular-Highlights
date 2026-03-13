import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepo.models.partialconv2d import PartialConv2d


class PConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        use_relu: bool = True,
    ):
        super().__init__()

        self.pconv = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,
            multi_channel=True,
            return_mask=True,
        )

        self.use_bn = use_bn
        self.use_relu = use_relu

        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _match_mask_channels(mask: torch.Tensor, channels: int) -> torch.Tensor:
        if mask.shape[1] == channels:
            return mask
        if mask.shape[1] == 1:
            return mask.expand(-1, channels, -1, -1)
        mask = mask.mean(dim=1, keepdim=True)
        return mask.expand(-1, channels, -1, -1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self._match_mask_channels(mask, x.shape[1]).to(dtype=x.dtype)
        x, mask = self.pconv(x, mask)

        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)

        return x, mask


class PConvBlock(nn.Module):
    """
    Two partial-conv layers, analogous to your original ConvBlock.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = PConvLayer(in_channels, out_channels)
        self.conv2 = PConvLayer(out_channels, out_channels)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.conv1(x, mask)
        x, mask = self.conv2(x, mask)
        return x, mask


class EncoderBlock(nn.Module):
    """
    Your original encoder idea, but mask-aware.
    Returns skip features and pooled features.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.conv1 = PConvBlock(in_channels, out_channels)
        self.conv2 = PConvBlock(out_channels, out_channels)
        self.downsample = nn.MaxPool2d(2, 2) if downsample else nn.Identity()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, mask = self.conv1(x, mask)
        x, mask = self.conv2(x, mask)

        skip_x = x
        skip_mask = mask

        x_down = self.downsample(x)
        mask_down = F.interpolate(mask, size=x_down.shape[-2:], mode="nearest")

        return x_down, mask_down, skip_x, skip_mask


class DecoderBlock(nn.Module):
    """
    Your original decoder idea, but mask-aware.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv1 = PConvBlock(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = PConvBlock(out_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        skip_x: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.upsample(x)
        mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")

        if x.shape[-2:] != skip_x.shape[-2:]:
            x = F.interpolate(x, size=skip_x.shape[-2:], mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, size=skip_x.shape[-2:], mode="nearest")

        x = torch.cat([x, skip_x], dim=1)
        mask = torch.cat([mask, skip_mask], dim=1)

        x, mask = self.conv1(x, mask)
        x, mask = self.conv2(x, mask)

        return x, mask


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, depth: int, Attention=None, num_heads: int = 4, context_channels: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.depth = depth
        self.channels = []

        ch_in = in_channels
        ch_out = base_channels

        for _ in range(depth):
            self.blocks.append(EncoderBlock(ch_in, ch_out))
            self.channels.append(ch_out)

            if Attention is None:
                self.attns.append(nn.Identity())
            elif Attention is TransformerEncoderCA:
                self.attns.append(
                    Attention(num_channels=ch_out, num_heads=num_heads, context_channels=context_channels)
                )
            else:
                self.attns.append(
                    Attention(num_channels=ch_out, num_heads=num_heads)
                )

            ch_in = ch_out
            ch_out *= 2

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: Optional[torch.Tensor] = None):
        skips = []

        for block, attn in zip(self.blocks, self.attns):
            x, mask, skip_x, skip_mask = block(x, mask)

            if isinstance(attn, TransformerEncoderCA):
                x = attn(x, context)
            else:
                x = attn(x)

            skips.append((skip_x, skip_mask))

        return x, mask, skips


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, length: int, Transformer=None, num_heads: int = 4, context_channels: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        ch_in = in_channels
        for _ in range(length):
            self.blocks.append(PConvBlock(ch_in, out_channels))

            if Transformer is None:
                self.attns.append(nn.Identity())
            elif Transformer is TransformerEncoderCA:
                self.attns.append(
                    Transformer(num_channels=out_channels, num_heads=num_heads, context_channels=context_channels)
                )
            else:
                self.attns.append(
                    Transformer(num_channels=out_channels, num_heads=num_heads)
                )

            ch_in = out_channels

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: Optional[torch.Tensor] = None):
        for block, attn in zip(self.blocks, self.attns):
            x, mask = block(x, mask)

            if isinstance(attn, TransformerEncoderCA):
                x = attn(x, context)
            else:
                x = attn(x)

        return x, mask


class Decoder(nn.Module):
    def __init__(self, encoder_channels, bottleneck_channels, Attention=None, num_heads: int = 4, context_channels: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        in_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            out_channels = skip_channels
            self.blocks.append(DecoderBlock(in_channels, skip_channels, out_channels))

            if Attention is None:
                self.attns.append(nn.Identity())
            elif Attention is TransformerEncoderCA:
                self.attns.append(
                    Attention(num_channels=out_channels, num_heads=num_heads, context_channels=context_channels)
                )
            else:
                self.attns.append(
                    Attention(num_channels=out_channels, num_heads=num_heads)
                )

            in_channels = out_channels

    def forward(self, x: torch.Tensor, mask: torch.Tensor, skips, context: Optional[torch.Tensor] = None):
        skips = list(skips)

        for block, attn in zip(self.blocks, self.attns):
            skip_x, skip_mask = skips.pop()
            x, mask = block(x, mask, skip_x, skip_mask)

            if isinstance(attn, TransformerEncoderCA):
                x = attn(x, context)
            else:
                x = attn(x)

        return x, mask


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        return self.dropout(self.pos_encoding[t].squeeze(1))


class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, num_heads: int = 4, max_tokens: int = 1024):
        super().__init__()
        self.max_tokens = max_tokens
        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(num_channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        if N > self.max_tokens:
            return x

        x = x.reshape(B, C, N).permute(0, 2, 1)
        x_ln = self.ln(x)
        attn_output, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        x = attn_output + x
        x = self.ff_self(x) + x
        return x.permute(0, 2, 1).reshape(B, C, H, W)


class TransformerEncoderCA(nn.Module):
    def __init__(self, num_channels: int, num_heads: int = 4, context_channels: int = 1, max_tokens: int = 1024):
        super().__init__()
        self.max_tokens = max_tokens
        self.context_proj = nn.Conv2d(context_channels, num_channels, kernel_size=1)
        self.q_ln = nn.LayerNorm(num_channels)
        self.kv_ln = nn.LayerNorm(num_channels)
        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        if N > self.max_tokens:
            return x

        context = self.context_proj(context)
        Bc, Cc, Hc, Wc = context.shape
        if B != Bc or C != Cc:
            raise ValueError(f"Shape mismatch: x={tuple(x.shape)}, context={tuple(context.shape)}")

        x_seq = x.reshape(B, C, N).permute(0, 2, 1)
        ctx_seq = context.reshape(B, C, Hc * Wc).permute(0, 2, 1)

        q = self.q_ln(x_seq)
        kv = self.kv_ln(ctx_seq)

        attn_output, _ = self.mha(query=q, key=kv, value=kv)
        x_seq = x_seq + attn_output
        x_seq = x_seq + self.ff_self(x_seq)

        return x_seq.permute(0, 2, 1).reshape(B, C, H, W)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        depth: int,
        out_channels: int = 3,
        Attention=None,
        num_heads: int = 4,
        context_channels: int = 1,
        bottleneck_length: int = 2,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            Attention=Attention,
            num_heads=num_heads,
            context_channels=context_channels,
        )

        encoder_channels = self.encoder.channels
        bottleneck_in = encoder_channels[-1]
        bottleneck_out = bottleneck_in * 2

        self.bottleneck = Bottleneck(
            in_channels=bottleneck_in,
            out_channels=bottleneck_out,
            length=bottleneck_length,
            Transformer=Attention,
            num_heads=num_heads,
            context_channels=context_channels,
        )

        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            bottleneck_channels=bottleneck_out,
            Attention=Attention,
            num_heads=num_heads,
            context_channels=context_channels,
        )

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: Optional[torch.Tensor] = None):
        x, mask, skips = self.encoder(x, mask, context)
        x, mask = self.bottleneck(x, mask, context)
        x, mask = self.decoder(x, mask, skips, context)
        return self.final_conv(x)


if __name__ == "__main__":
    modelA = UNet(
        in_channels=3,
        base_channels=64,
        depth=4,
        out_channels=3,
        Attention=None,
    )

    modelB = UNet(
        in_channels=3,
        base_channels=64,
        depth=4,
        out_channels=3,
        Attention=TransformerEncoderSA,
        num_heads=4,
    )

    modelC = UNet(
        in_channels=3,
        base_channels=64,
        depth=4,
        out_channels=3,
        Attention=TransformerEncoderCA,
        context_channels=1,
        num_heads=4,
    )

    x = torch.randn(2, 3, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    context = torch.randn(2, 1, 64, 64)

    yA = modelA(x, mask)
    print("Model A:", yA.shape)

    yB = modelB(x, mask, context=context)
    print("Model B:", yB.shape)

    yC = modelC(x, mask, context=context)
    print("Model C:", yC.shape)