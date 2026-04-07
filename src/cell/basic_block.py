# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
r"""Some basic network blocks."""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import List
from collections import OrderedDict

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PatchEmbedding(nn.Module):
    def __init__(self, channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.channels = channels
        self.embed_dim = embed_dim
        self.patch_size_1, self.patch_size_2 = patch_size
        self.proj = nn.Linear(self.channels * self.patch_size_1 * self.patch_size_2, self.embed_dim)

    def forward(self, x):
        # x: (batch_size, channels, time, space)
        batch_size, channels, time, space = x.shape
        # Reshape and permute to (batch_size, num_patches, patch_size * channels)
        x = x.unfold(2, self.patch_size_1, self.patch_size_1).unfold(3, self.patch_size_2, self.patch_size_2)
        x = x.contiguous().view(batch_size, channels, -1, self.patch_size_1 * self.patch_size_2)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, channels * self.patch_size_1 * self.patch_size_2)
        # Linear projection
        x = self.proj(x)
        return x

class MLP(nn.Module):


    def __init__(self, layer_widths: List[int], activation: str = 'ReLU', initialize: str = 'kaiming') -> None:
        super().__init__()

        if len(layer_widths) < 2:
            raise ValueError("至少需要指定输入和输出维度。")

        layers = []
        for i in range(len(layer_widths) - 1):
            layer = nn.Linear(layer_widths[i], layer_widths[i+1])
            
            # 初始化权重
            if initialize.lower() == 'kaiming':
                init.kaiming_uniform_(layer.weight, nonlinearity='relu' if activation.lower() == 'relu' else 'tanh')
                init.zeros_(layer.bias)
            elif initialize.lower() == 'uniform':
                init.uniform_(layer.weight, -1/layer_widths[i]**0.5, 1/layer_widths[i]**0.5)
                init.uniform_(layer.bias, -1/layer_widths[i]**0.5, 1/layer_widths[i]**0.5)
            else:
                raise ValueError(f"不支持的初始化方法: {initialize}")
            
            layers.append(layer)
            
            if i < len(layer_widths) - 2:  # 不在最后一层添加激活函数
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif activation.lower() == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"不支持的激活函数: {activation}")

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class FNO_encoder(nn.Module):
    def __init__(self, in_channels, lifted_channels, modes1, modes2, device='cuda:0'):
        super(FNO_encoder, self).__init__()
        self.lifted_channels = lifted_channels
        self.lift = nn.Linear(in_channels, lifted_channels)
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1/lifted_channels
        self.weights = nn.Parameter(self.scale * torch.rand(lifted_channels, 2 * modes1, modes2, dtype=torch.cfloat))
        # self.weights_2 = nn.Parameter(self.scale * torch.rand(lifted_channels, 2 * modes1, modes2, dtype=torch.cfloat))
    def forward(self, x):
        batchsize, height, width = x.shape
        x = x.view(batchsize, height, width, 1)
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        truncate_size = (batchsize, self.lifted_channels, 2 * self.modes1, self.modes2)
        low_freq_fft = torch.zeros(truncate_size,dtype=x_ft.dtype,device=x_ft.device)
        low_freq_fft[:, :, :self.modes1, :self.modes2] = x_ft[:, :, :self.modes1, :self.modes2]
        low_freq_fft[:, :, -self.modes1:, :self.modes2] = x_ft[:, :, -self.modes1:, :self.modes2]
        low_freq_fft = low_freq_fft * self.weights
        # low_freq_fft = low_freq_fft * self.weights_2
        return low_freq_fft
    
class FNO_decoder(nn.Module):
    def __init__(self, lifted_channels, out_channels, height, width, modes1, modes2):
        super(FNO_decoder, self).__init__()
        self.height = height
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1/lifted_channels
        self.weights = nn.Parameter(self.scale * torch.rand(lifted_channels, 2 * modes1, modes2, dtype=torch.cfloat))
        self.project = nn.Linear(lifted_channels, out_channels)
    def forward(self, x_ft, pr = False):
        x_ft = x_ft * self.weights
        batch_size, channels= (x_ft.shape[0],x_ft.shape[1])
        modes1 = self.modes1
        modes2 = self.modes2
        x = torch.zeros(batch_size, channels, self.height, self.width, dtype=torch.cfloat, device=x_ft.device)
        x[:, :, :modes1, :modes2] = x_ft[:, :, :modes1, :modes2]
        x[:, :, -modes1:, :modes2] = x_ft[:, :, -modes1:, :modes2]
        x = torch.fft.irfftn(x, s=(self.height, self.width), dim=(-2, -1))
        # xa = x.clone()
        x = x.permute(0, 2, 3, 1)
        x = self.project(x)
        x = x.view(batch_size, self.height, self.width)
        # if pr:
        #     return x, xa
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class VisionTransformer(nn.Module):
    def __init__(self, in_channels, input_resolution, patch_height, patch_width, width, layers, heads, output_dim):
        super().__init__()
        self.height_resolution, self.width_resolution = input_resolution
        self.total_resolution = self.height_resolution * self.width_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width), bias=False)
        self.scale = width ** -0.5
        self.class_embedding = nn.Parameter(self.scale * torch.randn(width))
        self.patch_scale = patch_height * patch_width
        self.positional_embedding = nn.Parameter(self.scale * torch.randn(int(self.total_resolution/self.patch_scale) + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(self.scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND [grid ** 2 + 1, *, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes[0]
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
    

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x ,y), (in_channel, out_channel, x, y)  -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=(-1,-2))

        # Multiply relevant Fourier modes
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[..., :self.modes1, :self.modes2], self.weights)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class FNO1d_layer(nn.Module):
    def __init__(self, modes, width, layers_widths):
        super(FNO1d_layer, self).__init__()
        self.modes1 = modes
        self.width = width
        self.conv = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp = MLP(layers_widths)
        self.w = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x):
        x0 = x.permute(0, 2, 1)
        x1 = self.conv(x0)
        x1 = x1.permute(0, 2, 1)
        x1 = self.mlp(x1)
        x2 = self.w(x0)
        x2 = x2.permute(0, 2, 1)
        x = x1 + x2
        x = F.gelu(x)
        return x
    
class FNO2d_layer(nn.Module):
    def __init__(self, modes, width, layers_widths):
        super(FNO2d_layer, self).__init__()
        self.modes1= modes[0]
        self.modes2= modes[1]
        self.width = width
        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp = MLP(layers_widths)
        self.w = nn.Conv2d(self.width, self.width, 1)

    def forward(self, x):
        x0 = x.permute(0, 3, 1, 2)
        x1 = self.conv(x0)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.mlp(x1)
        x2 = self.w(x0)
        x2 = x2.permute(0, 2, 3, 1)
        x = x1 + x2
        x = F.gelu(x)
        return x

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, track_running_stats=True):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AffineCouplingLayer(nn.Module):
    def __init__(self, net: nn.Module, tau=0.05):
        super(AffineCouplingLayer, self).__init__()
        self.net = net
        self.sp = nn.Softplus(beta=tau)
    def forward(self, v1, v2):
        v2_SL = self.sp(self.net(v2))
        v1 = v1 * v2_SL
        v1_SL = self.sp(self.net(v1))
        v2 = v1_SL * v2
        return v1, v2

    def inverse(self, v1, v2):
        v1_SL = self.sp(self.net(v1))
        v2 = v2 / v1_SL
        v2_SL = self.sp(self.net(v2))
        v1 = v1 / v2_SL
        return v1, v2
    
class RealNVP(nn.Module):
    def __init__(self, num_blocks, modes, width, layers_widths, dim, tau=0.05):
        super(RealNVP, self).__init__()
        if dim == 1:
            self.plug_net = nn.ModuleList([FNO1d_layer(modes,width,layers_widths) for _ in range(num_blocks)])
        elif dim == 2:
            self.plug_net = nn.ModuleList([FNO2d_layer(modes,width,layers_widths) for _ in range(num_blocks)])
        self.layers = nn.ModuleList([AffineCouplingLayer(block,tau) for block in self.plug_net])

    def forward(self, v1, v2):
        for layer in self.layers:
            v1, v2 = layer(v1,v2)
        return v1, v2

    def inverse(self, v1, v2):
        for layer in reversed(self.layers):
            v1, v2 = layer.inverse(v1, v2)
        return v1, v2

class INR_block(nn.Module):
    def __init__(self, coord_dim, embed_dim, scale_dims, shift_dims):
        super(INR_block, self).__init__()
        self.scale_mlp = MLP(scale_dims)
        self.shift_mlp = MLP(shift_dims)
        self.lifting_layer = nn.Linear(coord_dim, embed_dim)
        self.progress_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, coords, features_mode, x):
        features_mode = features_mode.transpose(0,1)
        coords_features = self.lifting_layer(coords)
        scale = self.scale_mlp(features_mode)
        shift = self.shift_mlp(features_mode)
        # progress x
        x = x * coords_features
        x = self.progress_layer(x)
        # reshape x to fit scale and shift
        # broadcast scale and shift to x
        batch_size, *grid_shape, embed_dim = x.shape
        x = x.reshape(batch_size, -1, embed_dim)
        x = scale * x + shift
        x = x.reshape(batch_size, *grid_shape, embed_dim)
        x = F.leaky_relu(x, negative_slope=0.2)
        return x
    
# class MultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.0):
#         super(MultiheadAttention, self).__init__()
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.dropout = nn.Dropout(dropout)
        
#         # Linear layers for query, key, and value
#         self.q_linear = nn.Linear(embed_dim, embed_dim)
#         self.k_linear = nn.Linear(embed_dim, embed_dim)
#         self.v_linear = nn.Linear(embed_dim, embed_dim)
        
#         # Output linear layer
#         self.out_linear = nn.Linear(embed_dim, embed_dim)
    
#     def scaled_dot_product_attention(self, q, k, v, mask=None):
#         scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
#         output = torch.matmul(attn_weights, v)
#         return output, attn_weights
    
#     def forward(self, query, key, value, mask=None):
#         batch_size = key.shape[0]
#         # Linear projections
#         q = self.q_linear(query)
#         k = self.k_linear(key)
#         v = self.v_linear(value)
#         # Reshape for multi-head attention
#         q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
#         # Apply scaled dot-product attention
#         attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
#         # Concatenate heads and put through final linear layer
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
#         output = self.out_linear(attn_output)
#         # 2024.11.15 14:12
#         output = output.view(batch_size, -1, self.embed_dim)
        
#         return output, attn_weights

# class TransformerLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, dim_feedforward=128, dropout=0.1):
#         super(TransformerLayer, self).__init__()
#         self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.linear1 = nn.Linear(embed_dim, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, embed_dim)

#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, src, src_mask=None):
#         src2, _ = self.self_attn(src, src, src, src_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

# class TransformerEncoder(nn.Module):
#     def __init__(self, embed_dim, num_heads, num_layers, dim_feedforward=2048, dropout=0.1, patch_size=(16, 16), channels=1, output_dim=128):
#         super(TransformerEncoder, self).__init__()
#         self.layers = nn.ModuleList([
#             TransformerLayer(embed_dim, num_heads, dim_feedforward, dropout)
#             for _ in range(num_layers)
#         ])
#         self.norm = nn.LayerNorm(embed_dim)
#         self.patch_embedding = PatchEmbedding(channels, patch_size, embed_dim)
#         self.fc = nn.Linear(embed_dim*patch_size[0]*patch_size[1], output_dim)
#     def forward(self, src, src_mask=None):
#         src = self.patch_embedding(src)
#         for layer in self.layers:
#             src = layer(src)
#         src = self.norm(src)
#         src = src.view(src.shape[0], -1)
#         src = self.fc(src)
#         return src


