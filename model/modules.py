"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from f5_tts.model.logger import Logger
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb
from f5_tts.model.utils import FLOPsCounter,CompressManager


# raw wav to mel spec


mel_basis_cache = {}
hann_window_cache = {}

# 原始音频波形转换为Mel频谱
def get_bigvgan_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=None,
    center=False,
):  # Copy from https://github.com/NVIDIA/BigVGAN/tree/main
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)  # TODO: why they need .float()?
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    ):
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram
        elif mel_spec_type == "bigvgan":
            self.extractor = get_bigvgan_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):  # noqa: F722
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


# rotary positional embedding related


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor | FastAttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only=None,
        block_id=None
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor
        self.block_id = block_id
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b n d"] = None,  # context c  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
        window_ratio = None
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope,block_id=self.block_id,window_ratio=window_ratio)
        else:
            return self.processor(self, x, mask=mask, rope=rope,block_id=self.block_id,window_ratio=window_ratio)



# Attention processor


class AttnProcessor:
    def __init__(self):
        self.flops_counter = FLOPsCounter() #! 计数器

    def _create_window_mask(self, seq_len, window_ratio=None):
        """创建滑动窗口掩码
        Args:
            seq_len: 序列长度
            window_ratio: 窗口比例,default 0.125
        """
        mask = None
        if window_ratio is None:
            return mask
        else:
            window_size = int(seq_len * window_ratio)
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[i, start:end] = True
            return mask

    # 结合传入掩码和窗口比得到最终掩码
    def _get_final_mask(self, mask, x, attn_heads, window_ratio=None):
        if window_ratio is not None:
            window_mask = self._create_window_mask(x.shape[1], window_ratio).to(x.device) # 创建窗口掩码 [seq_len, seq_len]
            window_mask = window_mask.unsqueeze(0).unsqueeze(0) # 扩展到 [1, 1, seq_len, seq_len]
            # 扩展到所有batch和head [batch_size, n_heads, seq_len, seq_len]
            window_mask = window_mask.expand(x.shape[0], attn_heads, x.shape[1], x.shape[1])
            
            if mask is not None:
                # 处理padding掩码 [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                attn_mask = mask.unsqueeze(1).unsqueeze(1) # 扩展到 [batch_size, 1, 1, seq_len]
                attn_mask = attn_mask.expand(x.shape[0], attn_heads, x.shape[1], x.shape[1])
                # 组合掩码
                final_mask = window_mask & attn_mask
            else:
                final_mask = window_mask
        elif mask is not None:
            # 原有的掩码处理逻辑
            attn_mask = mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            final_mask = attn_mask.expand(x.shape[0], attn_heads, x.shape[1], x.shape[1])
        else:
            final_mask = None

        return final_mask

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
        block_id=None,
        window_ratio=None
    ) -> torch.FloatTensor:
        attn_mask = self._get_final_mask(mask, x,attn.heads, window_ratio)
        batch_size = x.shape[0]

        # `sample` projections.
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        #! 统计投影层的FLOPS
        batch_size, seq_len, _ = x.shape
        self.flops_counter.add_linear_flops(attn.dim, attn.inner_dim, batch_size, seq_len)  # Q投影
        self.flops_counter.add_linear_flops(attn.dim, attn.inner_dim, batch_size, seq_len)  # K投影
        self.flops_counter.add_linear_flops(attn.dim, attn.inner_dim, batch_size, seq_len)  # V投影

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding

        #! 统计注意力计算的FLOPS
        self.flops_counter.add_attention_flops(query, key, value, attn.heads,window_ratio)

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)
        #-------------------------保存注意力权重------------------------------
        # logger = Logger()
        # # Compute the attention weights
        # logger.info(f'Compute the attention weights')
        
        # attn_weight = query @ key.transpose(-2, -1)  # [b, heads, n, n]
        # scale_factor = 1 / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32))
        # logger.info(f'Scale factor: {scale_factor}')

        # attn_weight = attn_weight * scale_factor  # scaled attention score

        # if attn_mask is not None:
        #     attn_weight += attn_mask

        # logger.info(f'Save attention weights')

        # # Save attention weights
        # attention_weights = torch.softmax(attn_weight, dim=-1)
        # output_dir = "./attention_weights"
        # # Save the attention weights to file
        # attention_file = os.path.join(output_dir, f"attn_weights_block_{block_id}.pt")
        # torch.save(attention_weights, attention_file)

        # logger.info(f'Attention weights saved to {attention_file}')
        #----------------------------------------------------------------------------
        # linear proj
        x = attn.to_out[0](x)
        #! 统计输出投影的FLOPS
        self.flops_counter.add_linear_flops(attn.inner_dim, attn.dim, batch_size, seq_len)
        
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x

# Fast Attention processor for FastAttn
# modified from diffusers/src/diffusers/models/attention_processor.py

class FastAttnProcessor:
    def __init__(self, window_size=256):
        self.window_size = window_size
        self.flops_counter = FLOPsCounter()
        
    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]
        
        # 投影到Q/K/V空间
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)
        
        # 统计投影层FLOPS
        batch_size, seq_len, _ = x.shape
        self.flops_counter.add_linear_flops(attn.dim, attn.inner_dim, batch_size, seq_len)  # Q
        self.flops_counter.add_linear_flops(attn.dim, attn.inner_dim, batch_size, seq_len)  # K
        self.flops_counter.add_linear_flops(attn.dim, attn.inner_dim, batch_size, seq_len)  # V
        
        # 应用旋转位置编码
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        
        # 重塑维度
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # 处理掩码
        if mask is not None:
            attn_mask = mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None
            
        # 1. 计算完整注意力
        full_attn = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # 2. 计算窗口注意力
        # 创建窗口掩码
        window_mask = self._create_window_mask(query.shape[-2], self.window_size).to(query.device)
        if attn_mask is not None:
            window_mask = window_mask & attn_mask
            
        window_attn = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=window_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # 3. 计算残差
        residual = full_attn - window_attn
        
        # 重塑回原始维度
        x = (window_attn + residual).transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)
        
        # 统计注意力计算的FLOPS
        self.flops_counter.add_attention_flops(query, key, value, attn.heads)
        
        # 输出投影
        x = attn.to_out[0](x)
        self.flops_counter.add_linear_flops(attn.inner_dim, attn.dim, batch_size, seq_len)
        
        # dropout
        x = attn.to_out[1](x)
        
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            
        return x
        
    def _create_window_mask(self, seq_len, window_size):
        """创建滑动窗口掩码"""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = True
        return mask

# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b nt d"] = None,  # context c, here text # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
        block_id=None
    ) -> torch.FloatTensor:
        residual = x

        batch_size = c.shape[0]

        # `sample` projections.
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections.
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # attention
        query = torch.cat([query, c_query], dim=1)
        key = torch.cat([key, c_key], dim=1)
        value = torch.cat([value, c_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )
        #-------------------------保存注意力权重------------------------------
        # # Compute the attention weights
        # attn_weight = query @ key.transpose(-2, -1)  # [b, heads, n, n]
        # scale_factor = 1 / torch.sqrt(query.shape[-1])
        # attn_weight = attn_weight * scale_factor  # scaled attention score

        # if attn_mask is not None:
        #     attn_weight += attn_mask

        # # Save attention weights
        # attention_weights = torch.softmax(attn_weight, dim=-1)
        # output_dir = "./attention_weights"
        # # Save the attention weights to file
        # attention_file = os.path.join(output_dir, f"joint_attn_weights_block_{block_id}.pt")
        # torch.save(attention_weights, attention_file)
        #----------------------------------------------------------------------------
        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c = attn.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c


# DiT Block


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1,block_id=None):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            block_id=block_id
        )
        self.block_id = block_id
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        self.compress_manager = CompressManager() #！记录压缩情况
        self.cur_step = 0 #!当前时间步

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # for key in self.compress_manager.compress_dict.keys():
        #     print(key)

        #！首先确认压缩方法
        method = self.compress_manager.get_method(self.cur_step)
        
        #! 若为时间步共享，直接读取上次输出
        if 'ast' in method:
            attn_output = self.compress_manager.cached_last_output if self.compress_manager.cached_last_output is not None else x

        #! 若为条件间的共享
        if 'asc' in method:
            # 先算无条件
            _,norm_uncond = norm.chunk(2, dim=0)
            attn_output_uncond = self.attn(x=norm_uncond, mask=mask, rope=rope)
            # 如果需要计算窗口残差
            if self.compress_manager.is_need_cal_res(self.cur_step):
                window_output_uncond = self.attn(x=norm_uncond, mask=mask, rope=rope,window_ratio=0.125)
                residual_uncond = attn_output_uncond - window_output_uncond
                self.compress_manager.cached_window_res = torch.cat([residual_uncond,residual_uncond], dim=0)
            attn_output = torch.cat([attn_output_uncond,attn_output_uncond], dim=0)

        #! 需要计算完整注意力的情况
        if 'wars' in method:
            #! 计算窗口注意力
            window_attn = self.attn(x=norm, mask=mask, rope=rope,window_ratio=0.125)
            #! 目前没有窗口残差缓存
            if self.compress_manager.cached_window_res is None:
                # 计算完整注意力，如果没有asc
                if 'asc' not in method:
                    attn_output = self.attn(x=norm, mask=mask, rope=rope)
                # 计算并缓存残差
                residual = attn_output - window_attn
                self.compress_manager.cached_window_res = residual
            else:
                # 使用缓存的残差
                attn_output = window_attn + self.compress_manager.cached_window_res
        if 'none' in method:
            attn_output = self.attn(x=norm, mask=mask, rope=rope)
            # 判断当前时间步是否需要计算残差
            if self.compress_manager.is_need_cal_res(self.cur_step):
                # 计算残差
                window_attn = self.attn(x=norm, mask=mask, rope=rope,window_ratio=0.125)
                residual = attn_output - window_attn
                # 缓存残差
                self.compress_manager.cached_window_res = residual
        
        #! 缓存ast
        self.compress_manager.cached_last_output = attn_output

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        #-----------------------------------
        return x


# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    context_pre_only: last layer only do prenorm + modulation cuz no more ffn
    """

    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, context_pre_only=False):
        super().__init__()

        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaLayerNormZero_Final(dim) if context_pre_only else AdaLayerNormZero(dim)
        self.attn_norm_x = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=dim,
            context_pre_only=context_pre_only,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None):  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)

        # process attention output for context c
        if self.context_pre_only:
            c = None
        else:  # if not last layer
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output

            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            c_ff_output = self.ff_c(norm_c)
            c = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c, x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time

