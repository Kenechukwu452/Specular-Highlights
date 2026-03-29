import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import Transformer

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        return self.dropout(self.pos_encoding[t].squeeze(1))
    
class ReduceBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1152, out_channels=1024, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(1024)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)     # (B, C=1152, S=256)
        x = self.conv(x)           # (B, 1024, 16)
        x = x.permute(0, 2, 1)     # (B, 16, 1024) ← correct format for LayerNorm
        x = self.norm(x)           # normalize over 1024 features
        x = self.activation(x)
        return x
class EmbedTime(nn.Module):
    def __init__(self, out_c, dim: int = 256):
        super().__init__()
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, out_c),
        )

    def forward(self, x, t):
        emb = self.emb_layer(t)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb

class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(num_channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        x_ln = self.ln(x)
        attn_output, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        x = attn_output + x
        x = self.ff_self(x) + x
        return x.permute(0, 2, 1).view(B, C, H, W)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
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
        x = torch.cat([x, skip], dim=1)
        convout = self.conv(x)
        return self.conv2(convout)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        b_neck= nn.ModuleList()
        for x in range(3):
            b_neck.append(ConvBlock(out_channels, out_channels))
            b_neck.append(TransformerEncoderSA(out_channels))
            x += 1
        self.conv2 = nn.Sequential(*b_neck)

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = self.conv2(conv_out)
        return conv_out

class UNetWithAttention(nn.Module):
    def __init__(self,noise_steps: int = 1000, time_dim: int = 256, tran: bool = False,depth: int = 2):
        """ Initialize UNet with Attention """  
        super().__init__()
        self.time_dim = time_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=noise_steps)
        """ Define UNet depth """
        self.in_channels = 6
        self.base_channels = 64
        self.depth = depth
    

        """ Set Down blocks """
        self.encoders = nn.ModuleList()
        self.time_embeds_down = nn.ModuleList()
        self.attns_down = nn.ModuleList()

        in_c = self.in_channels
        for i in range(self.depth):
            out_c = self.base_channels * (2 ** i)
            self.encoders.append(EncoderBlock(in_c, out_c))
            self.time_embeds_down.append(EmbedTime(out_c))
            if i < self.depth - 1:  
                self.attns_down.append(TransformerEncoderSA(out_c))
            in_c = out_c

        """ Create Bottleneck """
        self.bottleneck = Bottleneck(in_c,in_c * 2)
        bottleneck_out_c = in_c * 2

        """ setUp blocks """
        self.decoders = nn.ModuleList()
        self.time_embeds_up = nn.ModuleList()
        self.attns_up = nn.ModuleList()

        in_c = in_c if tran else bottleneck_out_c

        for i in reversed(range(self.depth)):
            out_c = self.base_channels * (2 ** i)
            self.decoders.append(DecoderBlock(in_c, out_c))
            self.time_embeds_up.append(EmbedTime(out_c))
            if i != 0:  
                self.attns_up.append(TransformerEncoderSA(out_c))
            in_c = out_c

        """ Final output layer """
        self.final_conv = nn.Conv2d(self.base_channels, 3, kernel_size=1)

    def forward(self, x, t: torch.LongTensor,c):
        t = self.pos_encoding(t)
        x = torch.cat([c,x], dim=1)
        skips = []
        """ Encoder """
        for i, encoder in enumerate(self.encoders):
            s, x = encoder(x)
            skips.append(s)
            x = self.time_embeds_down[i](x, t)
            if i < len(self.attns_down):
                x = self.attns_down[i](x)
        """ Bottleneck """
        x = self.bottleneck(x)

        """ Decoder """
        for i, decoder in enumerate(self.decoders):
            skip = skips.pop()
            x = decoder(x, skip)
            x = self.time_embeds_up[i](x, t)
            if i < len(self.attns_up):
                x = self.attns_up[i](x)
        """ Output """
        return self.final_conv(x)


    
class UNetWithTransformer(UNetWithAttention):
    def __init__(self, noise_steps: int = 1000, time_dim: int = 256,size=32,depth: int = 4):
        super().__init__(noise_steps=noise_steps, time_dim=time_dim, tran=True,depth=depth)
        
        # Set up Transformer bottleneck
        self.dit_channels = self.base_channels * 2 ** (self.depth - 1)  
        self.dit_patch_size = 2
        self.image_size = size  
        self.dit = Transformer.Transformer_S_2(
            input_size=self.image_size // (2 ** self.depth), 
            in_channels=self.dit_channels,
            learn_sigma=False
        )
        self.dit_proj_in = nn.Conv2d(self.dit_channels, self.dit.in_channels, kernel_size=1)
        self.dit_proj_out = nn.Conv2d(self.dit.out_channels, self.dit_channels, kernel_size=1)

    def forward(self, x, t_raw: torch.LongTensor, c):
        t = self.pos_encoding(t_raw)
        x = torch.cat([c,x], dim=1)
        skips = []

        # Encoder path
        for i, encoder in enumerate(self.encoders):
            s, x = encoder(x)
            skips.append(s)
            x = self.time_embeds_down[i](x, t)
            if i < len(self.attns_down):
                x = self.attns_down[i](x)

        # Bottleneck via Transformer
        x = self.dit_proj_in(x)
        x = self.dit(x, t=t_raw, y=c)  
        x = self.dit_proj_out(x)

        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip = skips.pop()
            x = decoder(x, skip)
            x = self.time_embeds_up[i](x, t)
            if i < len(self.attns_up):
                x = self.attns_up[i](x)

        return self.final_conv(x)


class UViT(Transformer.Transformer):
    def __init__(self,
                input_size=128,
                patch_size=4,
                in_channels=3,
                hidden_size=768,
                depth=8,
                num_heads=12,
                mlp_ratio=4.0,
                learn_sigma=False,
                conditioning_channels=3, 
                mids = 1):
        super().__init__(input_size=input_size,
                        patch_size=patch_size,
                        in_channels=   in_channels,
                        hidden_size=hidden_size,
                        depth=depth,
                        num_heads =    num_heads,
                        mlp_ratio= mlp_ratio,
                        learn_sigma=learn_sigma,
                        conditioning_channels=conditioning_channels)
        self.inblocks = nn.ModuleList([Transformer.TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(int(depth/2))
        ])
        
        self.mid_blocks = nn.ModuleList([
            Transformer.TransformerBlock(hidden_size,num_heads, mlp_ratio=mlp_ratio) for _ in range(mids)
        ])
        self.outblocks = nn.ModuleList([
            Transformer.TransformerBlock(hidden_size,num_heads, mlp_ratio=mlp_ratio) for _ in range(int(depth/2))
        ])
        self.reduce_block = ReduceBlock()
        self.skips =[]
        self.Linear = nn.Linear(2*hidden_size, hidden_size)
        self.mids = mids

    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y)    # (N, D)
        c = t + y                                # (N, D)

        #Input Blocks
        for block in self.inblocks:
            x = block(x, c)    # (N, T, D)
            self.skips.append(x)  
        
        #Mid Block
        for block in self.mid_blocks:
            x = block(x, c)    # (N, T, D)
          

        #Output Blocks
        for block in self.outblocks:
            x = self.Linear(torch.cat([x, self.skips.pop()], dim=-1))# Check Which dimension to concatenate
            x = block(x, c)
            
            #x= self.reduce_block(x)  # Reduce the sequence length
            #print(f"x.shape: {x.shape}") 
        
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

class UNetwithUViT(UNetWithAttention):
    def __init__(self, noise_steps: int = 1000, time_dim: int = 256,size=32,depth: int = 4):
        super().__init__(noise_steps=noise_steps, time_dim=time_dim, tran=True,depth=depth)
        
        # Set up Transformer bottleneck
        self.dit_channels = self.base_channels * 2 ** (self.depth - 1)  
        self.dit_patch_size = 2
        self.image_size = size  
        self.dit = UViT(depth=12, hidden_size=384, patch_size=8, num_heads=12,
            input_size=self.image_size // (2 ** self.depth),  
            in_channels=self.dit_channels,
            learn_sigma=False,
            mids = 1
        )
        self.dit_proj_in = nn.Conv2d(self.dit_channels, self.dit.in_channels, kernel_size=1)
        self.dit_proj_out = nn.Conv2d(self.dit.out_channels, self.dit_channels, kernel_size=1)

    def forward(self, x, t_raw: torch.LongTensor, c):
        t = self.pos_encoding(t_raw)
        x = torch.cat([c,x], dim=1)
        skips = []

        # Encoder 
        for i, encoder in enumerate(self.encoders):
            s, x = encoder(x)
            skips.append(s)
            x = self.time_embeds_down[i](x, t)
            if i < len(self.attns_down):
                x = self.attns_down[i](x)

        # Bottleneck Transformer
        x = self.dit_proj_in(x)
        x = self.dit(x, t=t_raw, y=c)  
        x = self.dit_proj_out(x)

        # Decoder 
        for i, decoder in enumerate(self.decoders):
            skip = skips.pop()
            x = decoder(x, skip)
            x = self.time_embeds_up[i](x, t)
            if i < len(self.attns_up):
                x = self.attns_up[i](x)

        return self.final_conv(x)
    

#################################################################################
#                                Standard Model and Names                       #
#################################################################################

def UNET(**kwargs):
    return UNetWithAttention(time_dim=256, depth= 2,**kwargs)

def DiT(**kwargs):
    return Transformer.Transformer_B_2(in_channels=3, learn_sigma = False,**kwargs)

def Flex(**kwargs):
    return UNetWithTransformer(time_dim=256, depth=2, **kwargs)

def UDiT(**kwargs):
    return UViT(in_channels=3, conditioning_channels=3, **kwargs)

def UTFLEX(**kwargs):
    return UNetwithUViT(time_dim=256, depth=2, **kwargs)