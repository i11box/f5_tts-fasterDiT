from __future__ import annotations

from functools import wraps
import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style

import logging

class FLOPsCounter:
    """用于统计模型计算量的工具类"""
    
    def __init__(self):
        self.total_flops = 0
        self.attention_flops = 0
        self.linear_flops = 0
        self.reset()
    
    def reset(self):
        """重置所有计数器"""
        self.total_flops = 0
        self.attention_flops = 0
        self.linear_flops = 0
    
    def add_attention_flops(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, heads: int,window_ratio=None):
        """添加注意力机制的FLOPS
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, dim]
            key: Key tensor of shape [batch_size, seq_len_k, dim]
            value: Value tensor of shape [batch_size, seq_len_v, dim]
            heads: Number of attention heads
        """
        flops = self.count_attention_flops(query, key, value, heads,window_ratio)
        self.attention_flops += flops
        self.total_flops += flops
    
    def add_linear_flops(
        self,
        input_size: int,
        output_size: int,
        batch_size: int,
        seq_len: int
        ):
        """添加线性层的FLOPS"""
        flops = self.count_linear_flops(input_size, output_size, batch_size, seq_len)
        self.linear_flops += flops
        self.total_flops += flops
    
    @staticmethod
    def count_attention_flops(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, heads: int,window_ratio=None):
        """统计注意力机制的FLOPS
        
        Args:
            query: Query tensor [batch_size, heads, seq_len, head_dim]
            key: Key tensor [batch_size, heads, seq_len, head_dim]
            value: Value tensor [batch_size, heads, seq_len, head_dim]
            heads: Number of attention heads
        
        Returns:
            total_flops: 注意力计算的总FLOPS数
        """
        # 获取张量形状
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # 1. Q @ K^T 的FLOPS
        if window_ratio is not None:
            # 窗口注意力：每个token只和窗口内的token计算注意力
            window_size = max(1, int(seq_len * window_ratio))
            if window_size % 2 == 0:
                window_size += 1
            qk_flops = batch_size * num_heads * seq_len * window_size * head_dim * 2
        else:
            # 普通注意力：每个token和所有token计算注意力
            qk_flops = batch_size * num_heads * seq_len * seq_len * head_dim * 2
            
        # 2. Softmax的FLOPS (exp + sum + div)
        if window_ratio is not None:
            softmax_flops = batch_size * num_heads * seq_len * window_size * 3
        else:
            softmax_flops = batch_size * num_heads * seq_len * seq_len * 3
            
        # 3. attention @ V 的FLOPS
        if window_ratio is not None:
            av_flops = batch_size * num_heads * seq_len * window_size * head_dim * 2
        else:
            av_flops = batch_size * num_heads * seq_len * seq_len * head_dim * 2
            
        total_flops = qk_flops + softmax_flops + av_flops
        
        return total_flops
    
    @staticmethod
    def count_linear_flops(input_size: int, output_size: int, batch_size: int, seq_len: int):
        """统计线性层的FLOPS
        
        Args:
            input_size: 输入特征维度
            output_size: 输出特征维度
            batch_size: batch大小
            seq_len: 序列长度
            
        Returns:
            flops: 线性层的FLOPS数
        """
        # 线性层的FLOPS = batch_size * seq_len * input_size * output_size
        return batch_size * seq_len * input_size * output_size
    
    def report(self):
        """生成FLOPS统计报告"""
        total_gflops = self.total_flops / 1e9
        attention_gflops = self.attention_flops / 1e9
        linear_gflops = self.linear_flops / 1e9
        
        return {
            "total_gflops": total_gflops,
            "attention_gflops": attention_gflops,
            "linear_gflops": linear_gflops,
            "attention_percentage": (attention_gflops / total_gflops * 100) if total_gflops > 0 else 0,
            "linear_percentage": (linear_gflops / total_gflops * 100) if total_gflops > 0 else 0
        }

#! 每个DiTBlock都有一个，用来存每个时间步的策略
class CompressManager:

    def __init__(self):
        self.compress_dict = {}
        self.strategy = ['ast','asc-wars','wars','asc']
        self.cached_last_output = None
        self.cached_uncond_output = None
        self.cached_window_res = None
        self.need_cal_window_res = {} # 记录需要计算窗口残差的时间步
    
    def reset(self): # 只重置缓存
        self.cached_last_output = None
        self.cached_uncond_output = None
        self.cached_window_res = None
    
    def calibrate_all_cal_res(self,calibrate_mode = True):
        self.need_cal_window_res = {
            t: True if calibrate_mode else False for t in self.compress_dict.keys()
        }
    
    def is_need_cal_res(self,t):
        '''
        判断当前时间步是否需要计算窗口残差
        '''
        return self.need_cal_window_res.get(f'{t.item():.3f}', False)
    
    def record(self, strategy,t):
        """
        单个块内，记录各个时间步采用策略
        """
        self.compress_dict.update({f'{t.item():.3f}':strategy})
        
    def get_method(self, t):
        """
        获取指定时间步的压缩策略
        """
        return self.compress_dict.get(f'{t.item():.3f}', 'none')
    
    def get_need_cal_window_res(self):
        """
        双指针计算需要计算窗口残差的时间步
        - i指针指向当前检查的none策略
        - j指针向后扫描寻找wars或下一个none
        """        
        if not self.compress_dict:
            return self.need_cal_window_res
        
        steps = sorted(float(step) for step in self.compress_dict.keys())
        
        i = 0
        while i < len(steps):
            current_strategy = self.compress_dict[f'{steps[i]:.3f}']
            
            # none和只计算asc的情况下都有可能需要计算窗口残差
            if 'none' in current_strategy or 'asc' == current_strategy:
                j = i + 1
                while j < len(steps):
                    next_strategy = self.compress_dict[f'{steps[j]:.3f}']
                    if 'wars' in next_strategy:
                        self.need_cal_window_res[f'{steps[i]:.3f}'] = True
                        # 找到wars后，直接跳到下一个位置继续搜索
                        break
                    if 'none' in next_strategy:
                        # 找到下一个none，从这里开始新的搜索
                        break
                    j += 1
                # j之前的都已经处理过了
                i = j
            else:
                i += 1
        
        return self.need_cal_window_res

# seed everything

def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    god_knows_why_en_testset_contains_zh_quote = str.maketrans(
        {"“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # in case librispeech (orig no-pc) test-clean
    custom_trans = str.maketrans({";": ","})  # add custom trans here, to address oov
    for text in text_list:
        char_list = []
        text = text.translate(god_knows_why_en_testset_contains_zh_quote)
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure chinese characters
                seg = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in seg:
                    if c not in "。，、；：？！《》【】—…":
                        char_list.append(" ")
                    char_list.append(c)
            else:  # if mixed chinese characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    else:
                        if c not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:  # if is zh punc
                            char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 日志装饰器
def log_method_call(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 记录方法调用的日志
        logging.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(self, *args, **kwargs)  # 调用原始方法
            logging.debug(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            raise e  # 抛出异常，以便继续处理
    return wrapper