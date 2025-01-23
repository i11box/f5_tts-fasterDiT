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
import json
import os

from datetime import datetime

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
            text[1:] = 0.0 # 这里只有后一半进行丢弃，模拟无条件生成
            # text = torch.zeros_like(text)

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
            cond[1:] = 0.0 # 这里只有后一半进行丢弃，模拟无条件生成
            # cond = torch.zeros_like(cond)

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
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

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
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # 由于ASC的加入，输入变成了两倍，设置时间步的时候需要分开
        t_single,_ = torch.chunk(time, 2)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            block.cur_step = t_single
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)       

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

    def collect_compression_strategies(self):
        """收集所有块的压缩策略
        
        Returns:
            dict: 包含所有块的压缩策略信息
            {
                block_id: {timestep: strategy}
            }
        """
        strategies = {}
        
        # 收集DiT的策略
        for block in self.transformer_blocks:
            if block.block_id is not None:
                strategies[block.block_id] = block.compress_manager.compress_dict

                    
        return strategies
        
    def print_compression_summary(self):
        """打印压缩策略的统计信息"""
        strategies = self.collect_compression_strategies()
                
        print(f"\n{dit_type.upper()} DiT压缩策略统计:")
        total_steps = 0
        strategy_counts = {'ast': 0, 'asc': 0, 'wars': 0, 'none': 0,'asc-wars':0}
        
        for block_id, block_strategies in strategies.items():
            print(f"\nBlock {block_id}:")
            block_counts = {'ast': 0, 'asc': 0, 'wars': 0, 'none': 0,'asc-wars':0}
            
            for t, strategy in block_strategies.items():
                block_counts[strategy] += 1
                strategy_counts[strategy] += 1
                total_steps += 1
            
            # 打印每个块的统计
            for strategy, count in block_counts.items():
                if count > 0:
                    percentage = count / len(block_strategies) * 100
                    print(f"  {strategy.upper()}: {count} steps ({percentage:.1f}%)")
        
        # 打印总体统计
        print(f"\n总体统计 (总时间步: {total_steps}):")
        for strategy, count in strategy_counts.items():
            if count > 0:
                percentage = count / total_steps * 100
                print(f"{strategy.upper()}: {count} steps ({percentage:.1f}%)")
                    
    def save_compression_strategies(self,file_name:str = 'method.json'):
        """保存压缩策略到文件
        
        Args:
            save_path: 保存路径，应以.json结尾
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        # 构建文件路径
        save_path = os.path.join(project_root, file_name)
        # 收集策略
        strategies = self.collect_compression_strategies()
        
        # 添加元信息
        save_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategies': strategies,
            'statistics': {}
        }
        
        # 计算统计信息
                
        total_steps = 0
        strategy_counts = {'ast': 0, 'asc': 0, 'wars': 0, 'none': 0,'asc-wars':0}
        
        for block_strategies in strategies.values():
            for strategy in block_strategies.values():
                strategy_counts[strategy] += 1
                total_steps += 1
        
        # 计算百分比
        percentages = {
            strategy: (count / total_steps * 100 if total_steps > 0 else 0)
            for strategy, count in strategy_counts.items()
        }
        
        save_data['statistics'] = {
            'total_steps': total_steps,
            'strategy_counts': strategy_counts,
            'strategy_percentages': percentages
        }
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
            
        print(f"压缩策略已保存到: {save_path}")
        
    def load_compression_strategies(self, load_path: str):
        """从文件加载压缩策略
        
        Args:
            load_path: 策略文件路径
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        # 构建文件路径
        load_path = os.path.join(project_root, load_path)
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        strategies = data['strategies']
        
        # 更新条件DiT的策略
        for block_id, block_strategies in strategies.items():
            for block in self.transformer_blocks:
                if block.block_id == int(block_id):  # JSON的键都是字符串
                    block.compress_manager.compress_dict = block_strategies
                        
        print(f"已从 {load_path} 加载压缩策略")
        print("\n策略统计信息:")
        stats = data['statistics']
        print(f"总时间步: {stats['total_steps']}")
        for strategy, percentage in stats['strategy_percentages'].items():
            count = stats['strategy_counts'][strategy]
            if count > 0:
                print(f"{strategy.upper()}: {count} steps ({percentage:.1f}%)")
    
    def set_all_block_id(self):
        cnt = 0
        for block in self.transformer_blocks:
            block.block_id = cnt
            cnt += 1
    
    def set_all_block_no_method(self):
        for block in self.transformer_blocks:
            block.compress_manager.compress_dict = {}
    
