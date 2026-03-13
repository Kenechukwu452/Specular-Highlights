import torch
import torch.nn as nn
import torch.nn.functional as F

from prepo.models.partialconv2d import PartialConv2d


class ConvolutionalShift(nn.Module):
    def __init__(self, align_corners: bool = True, eps: float = 1e-6):
        super().__init__()
        self.align_corners = align_corners
        self.eps = eps

    def forward(self, x: torch.Tensor, soft_mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        m = soft_mask.to(dtype=x.dtype)

        ys = torch.arange(1, H + 1, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        xs = torch.arange(1, W + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, W)

        denom = m.sum(dim=(2, 3)) + self.eps
        y_mean = (m * ys).sum(dim=(2, 3)) / denom
        x_mean = (m * xs).sum(dim=(2, 3)) / denom

        dy_pix = y_mean - (H + 1) / 2.0
        dx_pix = x_mean - (W + 1) / 2.0

        if self.align_corners:
            dy = dy_pix * (2.0 / max(H - 1, 1))
            dx = dx_pix * (2.0 / max(W - 1, 1))
        else:
            dy = dy_pix * (2.0 / H)
            dx = dx_pix * (2.0 / W)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)

        x_bc = x.reshape(B * C, 1, H, W)
        dx_bc = dx.reshape(B * C, 1, 1)
        dy_bc = dy.reshape(B * C, 1, 1)

        grid = base_grid.unsqueeze(0).repeat(B * C, 1, 1, 1).clone()
        grid[..., 0] = grid[..., 0] + dx_bc
        grid[..., 1] = grid[..., 1] + dy_bc

        x_shifted = F.grid_sample(
            x_bc,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )
        return x_shifted.reshape(B, C, H, W)


class MaskedAEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_conv_shift: bool = False,
    ):
        super().__init__()

        self.use_conv_shift = use_conv_shift

        self.pconv = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            multi_channel=True,
            return_mask=True,
        )

        self.shift = ConvolutionalShift(align_corners=True) if use_conv_shift else None
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _match_mask_channels(mask: torch.Tensor, channels: int) -> torch.Tensor:
        if mask.shape[1] == channels:
            return mask
        if mask.shape[1] == 1:
            return mask.expand(-1, channels, -1, -1)
        mask = mask.mean(dim=1, keepdim=True)
        return mask.expand(-1, channels, -1, -1)

    def forward(
        self, x: torch.Tensor, soft_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_mc = self._match_mask_channels(soft_mask, x.shape[1]).to(dtype=x.dtype)

        x = x * mask_mc

        if self.use_conv_shift:
            x = self.shift(x, mask_mc)

        x, updated_mask = self.pconv(x, mask_mc)
        x = self.bn(x)
        x = self.act(x)

        return x, updated_mask


class HighlightRemovalAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features: int = 64,
        depth: int = 4,
        max_features: int = 512,
        bottleneck_blocks: int = 2,
        use_conv_shift: bool = False,
    ):
        super().__init__()

        feats = [min(base_features * (2 ** i), max_features) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for i, ch in enumerate(feats):
            stride = 1 if i == 0 else 2
            self.encoders.append(
                MaskedAEBlock(
                    prev_ch,
                    ch,
                    stride=stride,
                    use_conv_shift=use_conv_shift,
                )
            )
            prev_ch = ch

        self.bottleneck = nn.ModuleList(
            [
                MaskedAEBlock(
                    feats[-1],
                    feats[-1],
                    stride=1,
                    use_conv_shift=use_conv_shift,
                )
                for _ in range(bottleneck_blocks)
            ]
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        rev_feats = feats[::-1]
        for i in range(len(rev_feats) - 1):
            in_ch = rev_feats[i]
            out_ch = rev_feats[i + 1]
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(
                MaskedAEBlock(
                    out_ch,
                    out_ch,
                    stride=1,
                    use_conv_shift=use_conv_shift,
                )
            )

        self.final_decoder = MaskedAEBlock(
            feats[0],
            feats[0],
            stride=1,
            use_conv_shift=use_conv_shift,
        )
        self.out = nn.Conv2d(feats[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _resize_mask(mask: torch.Tensor, hw) -> torch.Tensor:
        return F.interpolate(mask, size=hw, mode="nearest")

    def forward(self, x: torch.Tensor, soft_mask: torch.Tensor) -> torch.Tensor:
        h = x
        m = soft_mask.to(dtype=x.dtype)

        for enc in self.encoders:
            m = self._resize_mask(m, h.shape[-2:])
            h, m = enc(h, m)

        for block in self.bottleneck:
            m = self._resize_mask(m, h.shape[-2:])
            h, m = block(h, m)

        for up, dec in zip(self.upconvs, self.decoders):
            h = up(h)
            m = self._resize_mask(m, h.shape[-2:])
            h, m = dec(h, m)

        m = self._resize_mask(m, h.shape[-2:])
        h, m = self.final_decoder(h, m)

        y = self.out(h)
        return y
    
