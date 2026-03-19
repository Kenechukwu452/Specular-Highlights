import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = self.conv2(conv_out)
        pooled_out = self.pool(conv_out)
        return conv_out, pooled_out


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c * 2, out_c)
        self.conv2 = ConvBlock(out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        convout = self.conv(x)
        return self.conv2(convout)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim
        )

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        return self.dropout(self.pos_encoding[t].squeeze(1))


class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, num_heads: int = 4, max_tokens: int = 1024):
        super().__init__()
        self.max_tokens = max_tokens
        self.mha = nn.MultiheadAttention(
            embed_dim=num_channels,
            num_heads=num_heads,
            batch_first=True,
        )
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

        self.mha = nn.MultiheadAttention(
            embed_dim=num_channels,
            num_heads=num_heads,
            batch_first=True,
        )

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


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, Attention=None, num_heads=4, context_channels=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.depth = depth
        self.out_channels_per_stage = []

        for _ in range(depth):
            self.blocks.append(EncoderBlock(in_channels, out_channels))
            self.out_channels_per_stage.append(out_channels)

            if Attention is None:
                self.attns.append(nn.Identity())
            elif Attention is TransformerEncoderCA:
                self.attns.append(
                    Attention(out_channels, num_heads=num_heads, context_channels=context_channels)
                )
            else:
                self.attns.append(
                    Attention(out_channels, num_heads=num_heads)
                )

            in_channels = out_channels
            out_channels *= 2

    def forward(self, x, context=None):
        skips = []
        for i, block in enumerate(self.blocks):
            s, x = block(x)
            skips.append(s)

            attn = self.attns[i]
            if isinstance(attn, TransformerEncoderCA):
                if context is None:
                    raise ValueError("Cross-attention block requires context, but context=None was given.")
                x = attn(x, context)
            else:
                x = attn(x)

        return x, skips


class Decoder(nn.Module):
    def __init__(self, skip_channels, bottleneck_channels, Attention=None, num_heads=4, context_channels=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        in_c = bottleneck_channels
        for out_c in reversed(skip_channels):
            self.blocks.append(DecoderBlock(in_c, out_c))

            if Attention is None:
                self.attns.append(nn.Identity())
            elif Attention is TransformerEncoderCA:
                self.attns.append(
                    Attention(out_c, num_heads=num_heads, context_channels=context_channels)
                )
            else:
                self.attns.append(
                    Attention(out_c, num_heads=num_heads)
                )

            in_c = out_c

    def forward(self, x, skips, context=None):
        skips = list(skips)

        for i, block in enumerate(self.blocks):
            skip = skips.pop()
            x = block(x, skip)

            attn = self.attns[i]
            if isinstance(attn, TransformerEncoderCA):
                if context is None:
                    raise ValueError("Cross-attention block requires context, but context=None was given.")
                x = attn(x, context)
            else:
                x = attn(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, length, Attention=None, num_heads=4, context_channels=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        current_in = in_channels
        for _ in range(length):
            self.blocks.append(ConvBlock(current_in, out_channels))

            if Attention is None:
                self.attns.append(nn.Identity())
            elif Attention is TransformerEncoderCA:
                self.attns.append(
                    Attention(out_channels, num_heads=num_heads, context_channels=context_channels)
                )
            else:
                self.attns.append(
                    Attention(out_channels, num_heads=num_heads)
                )

            current_in = out_channels

    def forward(self, x, context=None):
        for block, attn in zip(self.blocks, self.attns):
            x = block(x)

            if isinstance(attn, TransformerEncoderCA):
                if context is None:
                    raise ValueError("Cross-attention block requires context, but context=None was given.")
                x = attn(x, context)
            else:
                x = attn(x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels,
        depth,
        Attention=None,
        OnlyonBottleneck=False,
        out_channels=3,
        num_heads=4,
        context_channels=1,
        bottleneck_length=2,
    ):
        super().__init__()

        enc_attention = None if OnlyonBottleneck else Attention
        dec_attention = None if OnlyonBottleneck else Attention
        bottleneck_attention = Attention

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=base_channels,
            depth=depth,
            Attention=enc_attention,
            num_heads=num_heads,
            context_channels=context_channels,
        )

        skip_channels = self.encoder.out_channels_per_stage
        bottleneck_in = skip_channels[-1]
        bottleneck_out = bottleneck_in * 2

        self.bottleneck = Bottleneck(
            in_channels=bottleneck_in,
            out_channels=bottleneck_out,
            length=bottleneck_length,
            Attention=bottleneck_attention,
            num_heads=num_heads,
            context_channels=context_channels,
        )

        self.decoder = Decoder(
            skip_channels=skip_channels,
            bottleneck_channels=bottleneck_out,
            Attention=dec_attention,
            num_heads=num_heads,
            context_channels=context_channels,
        )

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, context=None):
        x, skips = self.encoder(x, context)
        x = self.bottleneck(x, context)
        x = self.decoder(x, skips, context)
        return self.final_conv(x)


if __name__ == "__main__":
    modelA = UNet(
        in_channels=3,
        base_channels=64,
        depth=4,
        Attention=None,
    )

    modelB = UNet(
        in_channels=3,
        base_channels=64,
        depth=4,
        Attention=TransformerEncoderSA,
        num_heads=4,
        OnlyonBottleneck=False,
    )

    modelC = UNet(
        in_channels=3,
        base_channels=64,
        depth=4,
        Attention=TransformerEncoderCA,
        context_channels=1,
        num_heads=4,
        OnlyonBottleneck=True,
    )

    x = torch.randn(2, 3, 64, 64)
    context = torch.randn(2, 1, 64, 64)

    yA = modelA(x)
    print("Model A:", yA.shape)

    yB = modelB(x)
    print("Model B:", yB.shape)

    yC = modelC(x, context=context)
    print("Model C:", yC.shape)
