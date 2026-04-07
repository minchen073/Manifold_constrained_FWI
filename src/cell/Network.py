# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..torch_utils import persistence
from torch.nn.functional import silu
from .basic_block import *
from math import ceil
import os
from copy import deepcopy
#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=0.1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.
@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=0.1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

@persistence.persistent_class
class ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer1_config = config.layer1
        self.layer2_config = config.layer2
        self.layer3_config = config.layer3
        self.num_filters = config.num_filters
        self.feature_dim = config.feature_dim
        self.layer1 = Conv2DBlock(**self.layer1_config)
        self.layer2 = Conv2DBlock(**self.layer2_config)
        self.layer3 = Conv2DBlock(**self.layer3_config)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_filters, self.feature_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# class Conv2D_Block(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, track_running_stats=True):
#         super(Conv2D_Block, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats)
#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.avgpool(x)  # Flatten
#         return x
        
# @persistence.persistent_class
# class ConvEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.num_blocks = config.num_blocks
#         self.in_channels = config.in_channels
#         self.out_channels = config.out_channels
#         self.kernel_sizes = config.kernel_sizes
#         self.strides = config.strides
#         self.paddings = config.paddings
#         self.blocks = nn.ModuleList([Conv2D_Block(self.in_channels[i], self.out_channels[i], self.kernel_sizes[i], self.strides[i], self.paddings[i]) for i in range(self.num_blocks)])
#         self.MLP_layers = config.MLP_layers
#         self.projection = MLP(self.MLP_layers)
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         x = self.avgpool(x).view(x.size(0), -1)  # Flatten
#         x = self.projection(x)
#         return x

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        # 确保 num_groups 至少为 1，并且不超过 num_channels
        self.num_groups = max(1, min(num_groups, num_channels // min_channels_per_group, num_channels))
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):

    @staticmethod
    def setup_context(ctx, inputs, output):
        # 在这里保存任何需要在 backward 中使用的上下文信息
        pass

    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)  
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.up = up
        self.down = down
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-6)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.02))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

















# @persistence.persistent_class
# class ConditionEncoder(nn.Module):
#     """
#     条件编码器：基于InversionNet结构，最后一层输出cout个通道
#     """
#     def __init__(self, output_channels=192, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0):
#         super().__init__()
#         self.convblock1 = self._conv_block(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
#         self.convblock2_1 = self._conv_block(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
#         self.convblock2_2 = self._conv_block(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
#         self.convblock3_1 = self._conv_block(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
#         self.convblock3_2 = self._conv_block(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
#         self.convblock4_1 = self._conv_block(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
#         self.convblock4_2 = self._conv_block(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
#         self.convblock5_1 = self._conv_block(dim3, dim3, stride=2)
#         self.convblock5_2 = self._conv_block(dim3, dim3)
#         self.convblock6_1 = self._conv_block(dim3, dim4, stride=2)
#         self.convblock6_2 = self._conv_block(dim4, dim4)
#         self.convblock7_1 = self._conv_block(dim4, dim4, stride=2)
#         self.convblock7_2 = self._conv_block(dim4, dim4)
#         self.convblock8 = self._conv_block(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
#         self.deconv1_1 = self._deconv_block(dim5, dim5, kernel_size=5)
#         self.deconv1_2 = self._conv_block(dim5, dim5)
#         self.deconv2_1 = self._deconv_block(dim5, dim4, kernel_size=4, stride=2, padding=1)
#         self.deconv2_2 = self._conv_block(dim4, dim4)
#         self.deconv3_1 = self._deconv_block(dim4, dim3, kernel_size=4, stride=2, padding=1)
#         self.deconv3_2 = self._conv_block(dim3, dim3)
#         self.deconv4_1 = self._deconv_block(dim3, dim2, kernel_size=4, stride=2, padding=1)
#         self.deconv4_2 = self._conv_block(dim2, dim2)
#         self.deconv5_1 = self._deconv_block(dim2, dim1, kernel_size=4, stride=2, padding=1)
#         self.deconv5_2 = self._conv_block(dim1, dim1)
#         self.deconv6 = nn.Conv2d(dim1, output_channels, kernel_size=3, stride=1, padding=1)  # 修改输出通道数
#         self.bn = nn.BatchNorm2d(output_channels)  # 修改BatchNorm的通道数
#         self.tanh = nn.Tanh()

#     def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
    
#     def _deconv_block(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#     def forward(self, x):
#         """
#         Args:
#             x: [batch, 5, 1000, 70] 波场数据
#         Returns:
#             [batch, output_channels, 70, 70] 条件特征
#         """
#         x = self.convblock1(x)      # (batch, 32, 500, 70)
#         x = self.convblock2_1(x)    # (batch, 64, 250, 70)
#         x = self.convblock2_2(x)    # (batch, 64, 250, 70)
#         x = self.convblock3_1(x)    # (batch, 64, 125, 70)
#         x = self.convblock3_2(x)    # (batch, 64, 125, 70)
#         x = self.convblock4_1(x)    # (batch, 128, 63, 70)
#         x = self.convblock4_2(x)    # (batch, 128, 63, 70)
#         x = self.convblock5_1(x)    # (batch, 128, 32, 35)
#         x = self.convblock5_2(x)    # (batch, 128, 32, 35)
#         x = self.convblock6_1(x)    # (batch, 256, 16, 18)
#         x = self.convblock6_2(x)    # (batch, 256, 16, 18)
#         x = self.convblock7_1(x)    # (batch, 256, 8, 9)
#         x = self.convblock7_2(x)    # (batch, 256, 8, 9)
#         x = self.convblock8(x)      # (batch, 512, 1, 1)
        
#         x = self.deconv1_1(x)       # (batch, 512, 5, 5)
#         x = self.deconv1_2(x)       # (batch, 512, 5, 5)
#         x = self.deconv2_1(x)       # (batch, 256, 10, 10)
#         x = self.deconv2_2(x)       # (batch, 256, 10, 10)
#         x = self.deconv3_1(x)       # (batch, 128, 20, 20)
#         x = self.deconv3_2(x)       # (batch, 128, 20, 20)
#         x = self.deconv4_1(x)       # (batch, 64, 40, 40)
#         x = self.deconv4_2(x)       # (batch, 64, 40, 40)
#         x = self.deconv5_1(x)       # (batch, 32, 80, 80)
#         x = self.deconv5_2(x)       # (batch, 32, 80, 80)
        
#         x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (batch, 32, 70, 70)
        
#         x = self.deconv6(x)         # (batch, output_channels, 70, 70)
#         x = self.bn(x)
#         x = self.tanh(x)
#         return x
    
# #----------------------------------------------------------------------------
# # Reimplementation of the ADM architecture from the paper
# # "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# # original implementation by Dhariwal and Nichol, available at
# # https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        padding_resolution,                 # Padding resolution.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/12), init_bias=np.sqrt(1/12))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            # res = img_resolution >> level # 64/2^n
            res = padding_resolution >> level
            if level == 0:
                cin = cout # 1
                cout = model_channels * mult # 192 * [1,2,3,4]
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
                # self.enc['64x64_conv'] = Conv2d(in_channels=1, out_channels=192*1, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                # self.enc['32x32_down'] = UNetBlock(in_channels=192*1, out_channels=192*1, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            # res = img_resolution >> level
            res = padding_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if class_labels is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for name, block in self.enc.items():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)

        x = self.out_conv(silu(self.out_norm(x)))
        return x


# Import VelocityFlowUNet for flow matching
import sys
import os
flowmatching_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'flowmatching')
if flowmatching_path not in sys.path:
    sys.path.append(flowmatching_path)
try:
    from network import VelocityFlowUNet
    VELOCITY_FLOW_UNET_AVAILABLE = True
except ImportError:
    VELOCITY_FLOW_UNET_AVAILABLE = False
    print("Warning: VelocityFlowUNet not available, falling back to standard models")


class EDMToFlowAdapter(torch.nn.Module):
    """Adapter to make VelocityFlowUNet compatible with EDM interface."""

    def __init__(self, flow_net, sigma_min=0.002, sigma_max=80.0):
        super().__init__()
        self.flow_net = flow_net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, sigma, labels=None, class_labels=None, augment_labels=None):
        """
        Convert EDM sigma to flow matching time step.
        For flow matching: time goes from 0 to 1
        For EDM: sigma goes from sigma_max to sigma_min

        We'll map sigma to time as: time = 1 - (sigma - sigma_min) / (sigma_max - sigma_min)
        """
        # sigma can be a tensor with shape [batch_size, 1, 1, 1] or scalar
        if sigma.dim() > 0:
            # Normalize sigma to [0, 1] range
            sigma_norm = (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
            # Convert to flow matching time: when sigma is high (noisy), time should be low
            time = 1.0 - sigma_norm
        else:
            sigma_norm = (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
            time = 1.0 - sigma_norm

        # Ensure time is in valid range [0, 1]
        time = torch.clamp(time, 0.0, 1.0)

        return self.flow_net(x, time.squeeze())

    def round_sigma(self, sigma):
        """EDM compatibility method."""
        return torch.as_tensor(sigma)


@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        config,
        img_resolution,                     # Image resolution.
        padding_resolution,                 # Padding resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.config = config
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        # Handle VelocityFlowUNet specially
        if model_type == 'VelocityFlowUNet':
            if not VELOCITY_FLOW_UNET_AVAILABLE:
                raise ImportError("VelocityFlowUNet is not available. Please ensure flowmatching/network.py exists.")

            # Extract network configuration
            network_config = model_kwargs.get('network_config', {})

            # Create VelocityFlowUNet
            flow_net = VelocityFlowUNet(
                input_channels=network_config.get('input_channels', img_channels),
                input_height=network_config.get('input_height', img_resolution),
                padded_height=network_config.get('padded_height', 80),
                ch=network_config.get('ch', 32),
                ch_mult=network_config.get('ch_mult', [1, 2, 4, 8]),
                num_res_blocks=network_config.get('num_res_blocks', 4),
                attn_resolutions=network_config.get('attn_resolutions', []),
                dropout=network_config.get('dropout', 0.0),
                resamp_with_conv=network_config.get('resamp_with_conv', True)
            )

            # Wrap with EDM adapter for compatibility
            self.model = EDMToFlowAdapter(
                flow_net,
                sigma_min=sigma_min,
                sigma_max=sigma_max
            )
        else:
            # Use standard EDM models
            self.model = globals()[model_type](img_resolution=img_resolution, padding_resolution=padding_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def padding(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        return x
    
    def unpadding(self, x):
        x = x[:, :, 1:-1, 1:-1]
        return x
    
    def forward(self, x, sigma, class_labels=None, force_fp32=False, boundary_condition=False, **model_kwargs):
        x = x.to(torch.float32)

        # Check if using VelocityFlowUNet - it handles its own padding internally
        is_velocity_flow_unet = isinstance(self.model, EDMToFlowAdapter)

        if not is_velocity_flow_unet:
            x = self.padding(x)

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Standard EDM (teacher / sampling). For consistency distillation student/target, use
        # boundary_condition=True: Karras-style boundary so at sigma=sigma_min, D(x)=x (see paper/code).
        if boundary_condition:
            c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
            c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        else:
            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        if not is_velocity_flow_unet:
            D_x = self.unpadding(D_x)

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

    
@persistence.persistent_class
class ConditionalFusionDhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        padding_resolution,                 # Padding resolution.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        self.model_channels = model_channels
        emb_channels = model_channels * channel_mult_emb
        
        # 条件编码器
        self.condition_encoder = ConditionEncoder(
            output_channels=model_channels,
            dim1=32,
            dim2=64,
            dim3=128,
            dim4=256,
            dim5=512,
            sample_spatial=1.0
        )
        
        # 编码器初始化参数
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/12), init_bias=np.sqrt(1/12))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)
        
        # 时间嵌入 - 与原始DhariwalUNet保持一致
        self.map_noise = PositionalEmbedding(num_channels=model_channels)  # 移除endpoint=True
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None
        
        # 编码器
        
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = padding_resolution >> level  # 使用padding_resolution而不是img_resolution
            if level == 0:
                cin = cout  # 1
                cout = model_channels * mult  # 192
                # 添加初始卷积层来转换通道数
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                # 下采样层
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        
        # 解码器  
        skips = [block.out_channels for block in self.enc.values()]
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = padding_resolution >> level  # 使用padding_resolution而不是img_resolution
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        
        # 输出层
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)
        
    def forward(self, x, noise_labels, condition_data=None):
        """
        Args:
            x: [batch, in_channels, H, W] 噪声图像
            noise_labels: [batch,] 噪声水平
            class_labels: [batch, label_dim] 类别标签 (可选)
            condition_data: [batch, 5, 1000, 70] 波场条件数据
            augment_labels: [batch, augment_dim] 增强标签 (可选)
        Returns:
            [batch, out_channels, H, W] 去噪结果
        """
        # 编码条件数据
        if condition_data is not None:
            condition_features = self.condition_encoder(condition_data)  # [batch, model_channels, H, W]
        else:
            condition_features = None
        
        # 时间和标签嵌入 - 与原始DhariwalUNet保持一致  
        emb = self.map_noise(noise_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        emb = silu(emb)
        
        # 编码器前向传播
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # 解码器前向传播
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        
        # 条件融合与softmax
        if condition_features is not None:
            # 有条件：加上条件特征
            x = x + condition_features
        # 在通道维做softmax
        x = F.softmax(x, dim=1)
        # 输出层
        x = self.out_conv(silu(self.out_norm(x)))
        return x


@persistence.persistent_class
class ConditionalFusionEDMPrecond(torch.nn.Module):
    """
    带注意力融合的EDM预处理器
    """
    def __init__(self,
        config,
        img_resolution,                     # Image resolution.
        padding_resolution,                 # Padding resolution.
        img_channels,                       # Number of color channels.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.config = config
        self.img_resolution = img_resolution
        self.padding_resolution = padding_resolution
        self.img_channels = img_channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        # 创建网络
        self.model = ConditionalFusionDhariwalUNet(
            img_resolution=img_resolution,
            padding_resolution=padding_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            **model_kwargs
        )
    
    def padding(self, x):
        if x.shape[-1] < self.padding_resolution:
            pad_total = self.padding_resolution - x.shape[-1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_left, pad_right), mode='constant', value=0)
        return x
    
    def unpadding(self, x):
        if x.shape[-1] > self.img_resolution:
            crop_total = x.shape[-1] - self.img_resolution
            crop_left = crop_total // 2
            crop_right = crop_total - crop_left
            x = x[..., crop_left:x.shape[-2]-crop_right, crop_left:x.shape[-1]-crop_right]
        return x
    
    def forward(self, x, sigma, class_labels=None, condition_data=None, force_fp32=False, **model_kwargs):
        x = self.padding(x)
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        # 直接传递condition_data，无条件时为None
        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), condition_data=condition_data, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        D_x = self.unpadding(D_x)
        return D_x
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

class InversionNetForNoise(nn.Module):
    """
    基于InversionNet的结构，但为生成噪声进行了修改。
    关键改动：移除了最后的BatchNorm和Tanh激活函数，
    以允许网络输出无界且遵循高斯分布的噪声。
    """
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0):
        super(InversionNetForNoise, self).__init__()
        self.convblock1 = self._conv_block(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = self._conv_block(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = self._conv_block(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = self._conv_block(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = self._conv_block(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = self._conv_block(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = self._conv_block(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = self._conv_block(dim3, dim3, stride=2)
        self.convblock5_2 = self._conv_block(dim3, dim3)
        self.convblock6_1 = self._conv_block(dim3, dim4, stride=2)
        self.convblock6_2 = self._conv_block(dim4, dim4)
        self.convblock7_1 = self._conv_block(dim4, dim4, stride=2)
        self.convblock7_2 = self._conv_block(dim4, dim4)
        self.convblock8 = self._conv_block(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        self.deconv1_1 = self._deconv_block(dim5, dim5, kernel_size=5)
        self.deconv1_2 = self._conv_block(dim5, dim5)
        self.deconv2_1 = self._deconv_block(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = self._conv_block(dim4, dim4)
        self.deconv3_1 = self._deconv_block(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = self._conv_block(dim3, dim3)
        self.deconv4_1 = self._deconv_block(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = self._conv_block(dim2, dim2)
        self.deconv5_1 = self._deconv_block(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = self._conv_block(dim1, dim1)
        self.deconv6 = nn.Conv2d(dim1, 1, kernel_size=3, stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(1) # REMOVED
        # self.tanh = nn.Tanh() # REMOVED

    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _deconv_block(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5_1(x)
        x = self.convblock5_2(x)
        x = self.convblock6_1(x)
        x = self.convblock6_2(x)
        x = self.convblock7_1(x)
        x = self.convblock7_2(x)
        x = self.convblock8(x)
        
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = self.deconv2_1(x)
        x = self.deconv2_2(x)
        x = self.deconv3_1(x)
        x = self.deconv3_2(x)
        x = self.deconv4_1(x)
        x = self.deconv4_2(x)
        x = self.deconv5_1(x)
        x = self.deconv5_2(x)
        
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
        
        x = self.deconv6(x)
        # x = self.bn(x) # REMOVED
        # x = self.tanh(x) # REMOVED
        return x

class Wavefield2NoiseNet_Hybrid(nn.Module):
    """
    编码器采用InversionNet结构，将波场编码为[B, 512, 1, 1]，
    解码器采用全连接层，确保输出噪声的空间独立性。
    """
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0):
        super(Wavefield2NoiseNet_Hybrid, self).__init__()
        # 编码器部分
        self.convblock1 = self._conv_block(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = self._conv_block(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = self._conv_block(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = self._conv_block(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = self._conv_block(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = self._conv_block(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = self._conv_block(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = self._conv_block(dim3, dim3, stride=2)
        self.convblock5_2 = self._conv_block(dim3, dim3)
        self.convblock6_1 = self._conv_block(dim3, dim4, stride=2)
        self.convblock6_2 = self._conv_block(dim4, dim4)
        self.convblock7_1 = self._conv_block(dim4, dim4, stride=2)
        self.convblock7_2 = self._conv_block(dim4, dim4)
        self.convblock8 = self._conv_block(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(dim5, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 70 * 70),
        )
    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5_1(x)
        x = self.convblock5_2(x)
        x = self.convblock6_1(x)
        x = self.convblock6_2(x)
        x = self.convblock7_1(x)
        x = self.convblock7_2(x)
        x = self.convblock8(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        x = self.decoder(x)        # [B, 4900]
        x = x.view(x.size(0), 1, 70, 70)  # [B, 1, 70, 70]
        return x

@persistence.persistent_class
class ConsistencyModel(torch.nn.Module):
    """
    Consistency Model 结构，结构与 EDMPrecond 基本一致，
    但 skip-connection 系数采用 consistency model 论文的修正公式。
    """
    def __init__(self,
        config,
        img_resolution,                     # Image resolution.
        padding_resolution,                 # Padding resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.config = config
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](img_resolution=img_resolution, padding_resolution=padding_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def padding(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        return x
    
    def unpadding(self, x):
        x = x[:, :, 1:-1, 1:-1]
        return x
    
    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        x = self.padding(x)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        eps = self.sigma_min
        sigma_data = self.sigma_data
        # consistency model 论文的 skip-connection 公式
        c_skip = sigma_data ** 2 / ((sigma ** 2 - eps ** 2) + sigma_data ** 2)
        c_out = sigma_data * (sigma - eps) / (sigma_data ** 2 + sigma ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        D_x = self.unpadding(D_x)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

@persistence.persistent_class
class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.Identity,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride, 
            padding=padding, 
            bias=False,
        )
        self.norm = norm_layer(out_channels) if norm_layer != nn.Identity else nn.Identity()
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

@persistence.persistent_class
class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.Tanh(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1), 
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

@persistence.persistent_class
class Attention2d(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule2d(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

@persistence.persistent_class
class DecoderBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
    ):
        super().__init__()

        # Upsample block
        if upsample_mode == "pixelshuffle":
            from monai.networks.blocks import SubpixelUpsample
            self.upsample = SubpixelUpsample(
                spatial_dims=2,
                in_channels=in_channels,
                scale_factor=scale_factor,
            )
        else:
            from monai.networks.blocks import UpSample
            self.upsample = UpSample(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=in_channels,
                scale_factor=scale_factor,
                mode=upsample_mode,
            )

        if intermediate_conv:
            k = 3
            c = skip_channels if skip_channels != 0 else in_channels
            self.intermediate_conv = nn.Sequential(
                ConvBnAct2d(c, c, k, k//2),
                ConvBnAct2d(c, c, k, k//2),
            )
        else:
            self.intermediate_conv = None

        self.attention1 = Attention2d(
            name=attention_type, 
            in_channels=in_channels + skip_channels,
        )

        self.conv1 = ConvBnAct2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
        )

        self.conv2 = ConvBnAct2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
        )
        self.attention2 = Attention2d(
            name=attention_type, 
            in_channels=out_channels,
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.intermediate_conv is not None:
            if skip is not None:
                skip = self.intermediate_conv(skip)
            else:
                x = self.intermediate_conv(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

@persistence.persistent_class
class UnetDecoder2d(nn.Module):
    """
    Unet decoder.
    Source: https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self,
        encoder_channels: tuple[int],
        skip_channels: tuple[int] = None,
        decoder_channels: tuple = (256, 128, 64, 32),
        scale_factors: tuple = (2,2,2,2),
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = "scse",
        intermediate_conv: bool = True,
        upsample_mode: str = "pixelshuffle",
    ):
        super().__init__()
        
        if len(encoder_channels) == 4:
            decoder_channels = decoder_channels[1:]
        self.decoder_channels = decoder_channels
        
        if skip_channels is None:
            skip_channels = list(encoder_channels[1:]) + [0]

        # Build decoder blocks
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList()

        for i, (ic, sc, dc) in enumerate(zip(in_channels, skip_channels, decoder_channels)):
            self.blocks.append(
                DecoderBlock2d(
                    ic, sc, dc, 
                    norm_layer=norm_layer,
                    attention_type=attention_type,
                    intermediate_conv=intermediate_conv,
                    upsample_mode=upsample_mode,
                    scale_factor=scale_factors[i],
                )
            )

    def forward(self, feats: list[torch.Tensor]):
        res = [feats[0]]
        feats = feats[1:]

        # Decoder blocks
        for i, b in enumerate(self.blocks):
            skip = feats[i] if i < len(feats) else None
            res.append(
                b(res[-1], skip=skip),
            )
            
        return res

@persistence.persistent_class
class SegmentationHead2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: tuple[int] = (2,2),
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=kernel_size//2
        )
        from monai.networks.blocks import UpSample
        self.upsample = UpSample(
            spatial_dims=2,
            in_channels=out_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            mode=mode,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

@persistence.persistent_class
class Net(torch.nn.Module):
    """
    基于CAFormer backbone的U-Net架构，用于全波反演任务
    输入：波场数据 [batch, 5, 1000, 70]
    输出：速度场 [batch, 1, 70, 70]
    """
    def __init__(self, backbone="caformer_b36.sail_in22k_ft_in1k", pretrained=True):
        super().__init__()
        
        # 导入必要的模块
        try:
            import timm
        except ImportError:
            raise ImportError("需要安装timm库: pip install timm")
        
        # Encoder - CAFormer backbone
        self.backbone = timm.create_model(
            backbone,
            in_chans=5,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=0.0,
        )
        encoder_channels = [_["num_chs"] for _ in self.backbone.feature_info][::-1]

        # Decoder
        self.decoder = UnetDecoder2d(
            encoder_channels=encoder_channels,
        )

        self.seg_head = SegmentationHead2d(
            in_channels=self.decoder.decoder_channels[-1],
            out_channels=1,
            scale_factor=1,
        )
        
        self._update_stem(backbone)

    def _update_stem(self, backbone):
        m = self.backbone

        m.stem.conv.stride = (4, 1)
        m.stem.conv.padding = (0, 4)
        m.stages_0.downsample = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))
        m.stem = nn.Sequential(
            nn.ReflectionPad2d((0, 0, 78, 78)),
            m.stem,
        )

    def proc_flip(self, x_in):
        x_in = torch.flip(x_in, dims=[-3, -1])
        x = self.backbone(x_in)
        x = x[::-1]

        # Decoder
        x = self.decoder(x)
        x_seg = self.seg_head(x[-1])
        x_seg = x_seg[..., 1:-1, 1:-1]
        x_seg = torch.flip(x_seg, dims=[-1])
        x_seg = x_seg * 1500 + 3000
        return x_seg

    def forward(self, x):
        """
        Args:
            x: [batch, 5, 1000, 70] 波场数据
        Returns:
            [batch, 1, 70, 70] 速度场
        """
        # Encoder
        x_in = x
        x = self.backbone(x)
        x = x[::-1]

        # Decoder
        x = self.decoder(x)
        x_seg = self.seg_head(x[-1])
        x_seg = x_seg[..., 1:-1, 1:-1]
        x_seg = x_seg * 1500 + 3000
    
        if self.training:
            return x_seg
        else:
            p1 = self.proc_flip(x_in)
            x_seg = torch.mean(torch.stack([x_seg, p1]), dim=0)
            return x_seg

#----------------------------------------------------------------------------
# 从seismic-master仓库转移的网络模型类
# 基于CAFormer backbone的U-Net架构，用于全波反演任务
#----------------------------------------------------------------------------

@persistence.persistent_class
class ConvBnAct2d(torch.nn.Module):
    """卷积-批归一化-激活函数模块"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: torch.nn.Module = torch.nn.Identity,
        act_layer: torch.nn.Module = torch.nn.ReLU,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride, 
            padding=padding, 
            bias=False,
        )
        self.norm = norm_layer(out_channels) if norm_layer != torch.nn.Identity else torch.nn.Identity()
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


@persistence.persistent_class
class SCSEModule2d(torch.nn.Module):
    """SCSE注意力模块：通道注意力和空间注意力的结合"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels, in_channels // reduction, 1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(in_channels // reduction, in_channels, 1),
            torch.nn.Sigmoid(),
        )
        self.sSE = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 1, 1), 
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


@persistence.persistent_class
class Attention2d(torch.nn.Module):
    """2D注意力模块选择器"""
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = torch.nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule2d(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


@persistence.persistent_class
class DecoderBlock2d(torch.nn.Module):
    """U-Net解码器块"""
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: torch.nn.Module = torch.nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
    ):
        super().__init__()

        # 上采样块
        if upsample_mode == "pixelshuffle":
            try:
                from monai.networks.blocks import SubpixelUpsample
                self.upsample = SubpixelUpsample(
                    spatial_dims=2,
                    in_channels=in_channels,
                    scale_factor=scale_factor,
                )
            except ImportError:
                # 如果没有monai，使用标准的转置卷积
                self.upsample = torch.nn.ConvTranspose2d(
                    in_channels, in_channels, 
                    kernel_size=scale_factor, stride=scale_factor
                )
        else:
            try:
                from monai.networks.blocks import UpSample
                self.upsample = UpSample(
                    spatial_dims=2,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    scale_factor=scale_factor,
                    mode=upsample_mode,
                )
            except ImportError:
                # 如果没有monai，使用标准的转置卷积
                self.upsample = torch.nn.ConvTranspose2d(
                    in_channels, in_channels, 
                    kernel_size=scale_factor, stride=scale_factor
                )

        if intermediate_conv:
            k = 3
            c = skip_channels if skip_channels != 0 else in_channels
            self.intermediate_conv = torch.nn.Sequential(
                ConvBnAct2d(c, c, k, k//2),
                ConvBnAct2d(c, c, k, k//2),
            )
        else:
            self.intermediate_conv = None

        self.attention1 = Attention2d(
            name=attention_type, 
            in_channels=in_channels + skip_channels,
        )

        self.conv1 = ConvBnAct2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
        )

        self.conv2 = ConvBnAct2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
        )
        self.attention2 = Attention2d(
            name=attention_type, 
            in_channels=out_channels,
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.intermediate_conv is not None:
            if skip is not None:
                skip = self.intermediate_conv(skip)
            else:
                x = self.intermediate_conv(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


@persistence.persistent_class
class UnetDecoder2d(torch.nn.Module):
    """U-Net解码器"""
    def __init__(
        self,
        encoder_channels: tuple[int],
        skip_channels: tuple[int] = None,
        decoder_channels: tuple = (256, 128, 64, 32),
        scale_factors: tuple = (2,2,2,2),
        norm_layer: torch.nn.Module = torch.nn.Identity,
        attention_type: str = "scse",
        intermediate_conv: bool = True,
        upsample_mode: str = "pixelshuffle",
    ):
        super().__init__()
        
        if len(encoder_channels) == 4:
            decoder_channels = decoder_channels[1:]
        self.decoder_channels = decoder_channels
        
        if skip_channels is None:
            skip_channels = list(encoder_channels[1:]) + [0]

        # 构建解码器块
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = torch.nn.ModuleList()

        for i, (ic, sc, dc) in enumerate(zip(in_channels, skip_channels, decoder_channels)):
            self.blocks.append(
                DecoderBlock2d(
                    ic, sc, dc, 
                    norm_layer=norm_layer,
                    attention_type=attention_type,
                    intermediate_conv=intermediate_conv,
                    upsample_mode=upsample_mode,
                    scale_factor=scale_factors[i],
                )
            )

    def forward(self, feats: list[torch.Tensor]):
        res = [feats[0]]
        feats = feats[1:]

        # 解码器块
        for i, b in enumerate(self.blocks):
            skip = feats[i] if i < len(feats) else None
            res.append(
                b(res[-1], skip=skip),
            )
            
        return res


@persistence.persistent_class
class SegmentationHead2d(torch.nn.Module):
    """分割头模块"""
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: tuple[int] = (2,2),
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=kernel_size//2
        )
        try:
            from monai.networks.blocks import UpSample
            self.upsample = UpSample(
                spatial_dims=2,
                in_channels=out_channels,
                out_channels=out_channels,
                scale_factor=scale_factor,
                mode=mode,
            )
        except ImportError:
            # 如果没有monai，使用标准的转置卷积
            self.upsample = torch.nn.ConvTranspose2d(
                out_channels, out_channels,
                kernel_size=scale_factor, stride=scale_factor
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


@persistence.persistent_class
class SeismicNet(torch.nn.Module):
    """
    基于CAFormer backbone的U-Net架构，用于全波反演任务
    输入：波场数据 [batch, 5, 1000, 70]
    输出：速度场 [batch, 1, 70, 70]
    """
    def __init__(self, backbone="caformer_b36.sail_in22k_ft_in1k", pretrained=True):
        super().__init__()
        
        # 导入必要的模块
        try:
            import timm
        except ImportError:
            raise ImportError("需要安装timm库: pip install timm")
        
        # Encoder - CAFormer backbone
        self.backbone = timm.create_model(
            backbone,
            in_chans=5,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=0.0,
        )
        encoder_channels = [_["num_chs"] for _ in self.backbone.feature_info][::-1]

        # Decoder
        self.decoder = UnetDecoder2d(
            encoder_channels=encoder_channels,
        )

        self.seg_head = SegmentationHead2d(
            in_channels=self.decoder.decoder_channels[-1],
            out_channels=1,
            scale_factor=1,
        )
        
        self._update_stem(backbone)

    def _update_stem(self, backbone):
        """修改backbone的stem层以适应地震数据"""
        m = self.backbone

        m.stem.conv.stride = (4, 1)
        m.stem.conv.padding = (0, 4)
        m.stages_0.downsample = torch.nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))
        m.stem = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((0, 0, 78, 78)),
            m.stem,
        )

    def proc_flip(self, x_in):
        """处理翻转的数据增强"""
        x_in = torch.flip(x_in, dims=[-3, -1])
        x = self.backbone(x_in)
        x = x[::-1]

        # Decoder
        x = self.decoder(x)
        x_seg = self.seg_head(x[-1])
        x_seg = x_seg[..., 1:-1, 1:-1]
        x_seg = torch.flip(x_seg, dims=[-1])
        x_seg = x_seg * 1500 + 3000
        return x_seg

    def forward(self, x):
        """
        Args:
            x: [batch, 5, 1000, 70] 波场数据
        Returns:
            [batch, 1, 70, 70] 速度场
        """
        # Encoder
        x_in = x
        x = self.backbone(x)
        x = x[::-1]

        # Decoder
        x = self.decoder(x)
        x_seg = self.seg_head(x[-1])
        x_seg = x_seg[..., 1:-1, 1:-1]
        x_seg = x_seg * 1500 + 3000
    
        if self.training:
            return x_seg
        else:
            p1 = self.proc_flip(x_in)
            x_seg = torch.mean(torch.stack([x_seg, p1]), dim=0)
            return x_seg


@persistence.persistent_class
class EnsembleModel(torch.nn.Module):
    """集成多个模型的模块"""
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models).eval()

    def forward(self, x):
        output = None
        
        for m in self.models:
            logits = m(x)
            
            if output is None:
                output = logits
            else:
                output += logits
                
        output /= len(self.models)
        return output
