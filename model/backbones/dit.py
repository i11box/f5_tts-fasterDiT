"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""


from __future__ import annotations
# 导入日志类
from f5_tts.model.logger import Logger
import torch
from torch import nn
import torch.nn.functional as F


from x_transformers.x_transformers import RotaryEmbedding
from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AttnProcessor,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# Text embedding

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers) #! 输入维度增倍
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim) #！输入维度增倍

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout,block_id=i) for i in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio*2  # noqa: F722
        cond: float["b n d"],  # masked cond audio*2  # noqa: F722
        text: int["b nt"],  # text*2  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0]//2, x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        #! text变成两倍了，分别处理条件和无条件部分
        batch_size = text.shape[0] // 2
        text_uncond = self.text_embed(text[:batch_size], seq_len, drop_text=True)
        text_cond = self.text_embed(text[batch_size:], seq_len, drop_text=False)
        
        #! cond变成两倍了，分别处理条件和无条件部分
        cond_uncond = cond[:batch_size]
        cond_cond = cond[batch_size:]        
        
        #！x也要作拆分
        x_uncond, x_cond = x.chunk(2, dim=0)
        x_uncond = self.input_embed(x_uncond, cond_uncond, text_uncond, drop_audio_cond=True)
        x_cond = self.input_embed(x_cond, cond_cond, text_cond, drop_audio_cond=False)

        #！再合并
        x = torch.cat((x_uncond, x_cond), dim=0)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual_cond = x_cond
            residual_uncond = x_uncond

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x_uncond, x_cond = x.chunk(2, dim=0)
            x_cond = self.long_skip_connection(torch.cat((x_cond, residual_cond), dim=-1))
            x_uncond = self.long_skip_connection(torch.cat((x_uncond, residual_uncond), dim=-1))

        x_cond = self.norm_out(x_cond, t)
        x_cond = self.proj_out(x_cond)
        x_uncond = self.norm_out(x_uncond, t)
        x_uncond = self.proj_out(x_uncond)
        output = torch.cat((x_uncond, x_cond), dim=0)
        
        return output

    #! 汇报FLOPS数
    def report_flops(self):
        print("DiT.report_flops() called")
        total_stats = {
            "total_gflops": 0,
            "attention_gflops": 0,
            "linear_gflops": 0
        }
        
        print(f"Number of transformer blocks: {len(self.transformer_blocks)}")
        for i, block in enumerate(self.transformer_blocks):
            print(f"Processing block {i}")
            print(f"Block type: {type(block)}")
            print(f"Attention processor type: {type(block.attn.processor)}")
            
            if isinstance(block.attn.processor, AttnProcessor):
                print(f"Block {i} has AttnProcessor")
                block_stats = block.attn.processor.flops_counter.report()
                print(f"Block {i} stats: {block_stats}")
                for key in total_stats:
                    total_stats[key] += block_stats[key]
            else:
                print(f"Block {i} has no AttnProcessor")
                Logger.warning("no attnProcessor,maybe JointAttnProcessor")
        
        print(f"Final total stats: {total_stats}")
        return total_stats