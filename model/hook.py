from copy import deepcopy
import glob
import json
from multiprocessing import Process, Queue
import os
import types
from typing import Any, Optional, Tuple

from flash_attn import flash_attn_func
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from x_transformers.x_transformers import apply_rotary_pos_emb
from f5_tts.model.utils import FLOPsCounter,CompressManager
from f5_tts.model.modules import Attention
import math

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""
计算Attention的Flops
"""

def calculate_flops_hook(module, args, kwargs):
    # 从kwargs中获取hidden_states
    hidden_states = kwargs['x']
    batch_size, seq_len, dim = hidden_states.shape
    
    # 基础计算量：Q*K + Attention*V
    base_ops = seq_len * seq_len * module.heads * batch_size * dim // module.heads + seq_len * dim * batch_size * seq_len
    
    # 记录全精度计算量
    module.full_ops += base_ops
    
    # 获取当前方法和窗口大小
    method = module.steps_method[module.step]
    window_size = module.window_size * 2 if hasattr(module, 'window_size') else int(seq_len * module.window_ratio) * 2
    
    cond_op = deepcopy(base_ops)
    uncond_op = deepcopy(base_ops)
    op = [cond_op /2, uncond_op /2]
    
    # 根据不同方法计算实际计算量
    for i in range(2):
        if method[i] == "full_attention":
            if module.need_cache_residual[module.step][i]:
                op[i] *= 1 + window_size / seq_len
        elif method[i] == 'wars':
            op[i] *= window_size / seq_len
        elif method[i] == "ast" or method[i] == 'asc':
            op[i] = 0
    
    # 记录实际计算量
    module.efficient_ops += op[0] + op[1]

"""
计算raw output与efficient output之间的差距，使用默认值计算Loss
"""
def compression_loss(a, b, metric=""):
    ls = []
    for ai, bi in zip(a, b):
        if isinstance(ai, torch.Tensor):
            diff = (ai - bi) / (torch.max(ai, bi) + 1e-6)
            l = diff.abs().clip(0, 10).mean()
            ls.append(l)
    l = sum(ls) / len(ls)
    return l

"""
校准函数，使用贪心算出各个机制所对应的超参数
"""
def transformer_forward_pre_hook_for_calibration(model, args, kwargs):
    
    now_stepi = model.transformer_blocks[0].attn.step
    print(f"Calibration Step: {now_stepi}")

    # 为了避免在搜索candidate method时对cache内容产生改变，因此搜索时需要先关掉cache的开关
    for block in model.transformer_blocks:
        block.attn.forward = types.MethodType(efficient_attention_forward, block.attn)
        block.attn.need_cache_output[now_stepi] = [False,False]
        block.attn.need_cache_residual[now_stepi] = [False,False]

    # 总进度条
    total_blocks = len(model.transformer_blocks)
    method_candidates = [
        ['ast', 'ast'],
        ['ast', 'asc'],
        ['asc', 'ast'],
        ['ast', 'wars'],
        ['wars', 'ast'],
        ['wars','asc'],
        ['asc','wars'],
        ['wars', 'wars'],
        ['full_attention', 'ast'],
        ['ast','full_attention'],
        ['full_attention', 'asc'],
        ['asc', 'full_attention'],
        ['wars', 'full_attention'],
        ['full_attention', 'wars'],
    ]
    # method_candidates = [
    #     ['ast', 'ast'],
    #     ['wars','asc'],
    #     ['wars', 'wars'],
    #     ['full_attention', 'asc'],
    # ]
    # method_candidates = [
    #     ['ast', 'ast'],
    #     ['full_attention','ast'],
    #     ['ast','full_attention'],
    #     ['wars', 'wars'],
    #     ['full_attention','wars'],
    #     ['wars','full_attention'],
    # ]
    # method_candidates = [
    #     ['ast', 'ast'],
    #     ['wars', 'wars'],
    # ]
    
    if not hasattr(model, 'total_pbar'):
        total_steps = 32 * total_blocks * len(method_candidates)
        model.total_pbar = tqdm(
            total=total_steps,
            desc="总体校准进度",
            position=0
        )
    
    # 当前步进度条
    step_pbar = tqdm(
        total=total_blocks * len(method_candidates),
        desc=f"时间步 {now_stepi}/32",
        position=1,
        leave=False,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]',
        postfix=f"block 0/{total_blocks} method: initializing"
    )

    # 先走一遍得到full-attention的值
    raw_outputs = model.forward(*args, **kwargs)
    raw_output_cond,raw_output_uncond = raw_outputs.chunk(2,dim=0)
    raw_outputs = 2*raw_output_cond - raw_output_uncond
    for blocki, block in enumerate(model.transformer_blocks):
        if now_stepi == 0:
            continue
        # method的由强到弱
        selected_method = ['full_attention','full_attention'] # 第一个代表cond
        for method in method_candidates:
            step_pbar.set_postfix_str(f"block {blocki + 1}/{total_blocks} method: {method}")
            # print(f"Try###Block:{blocki} Step:{now_stepi} Method:{method}")
            block.attn.steps_method[now_stepi] = method

            for block_ in model.transformer_blocks:
                block_.attn.step = now_stepi
            efficient_outputs = model.forward(*args, **kwargs)
            efficient_output_cond,efficient_output_uncond = efficient_outputs.chunk(2,dim=0)
            efficient_outputs = 2*efficient_output_cond - efficient_output_uncond
            loss = compression_loss(raw_outputs, efficient_outputs)
            threshold = model.loss_thresholds[now_stepi][blocki]
            # print(f"Try### Block:{blocki} Step:{now_stepi} Method:{method} Loss:{loss} Threshold:{threshold}")

            if loss<threshold:
                remaining = len(method_candidates) - method_candidates.index(method)
                step_pbar.update(remaining)
                model.total_pbar.update(remaining)
                selected_method = method
                break
            
            step_pbar.update(1)
            model.total_pbar.update(1)
            
        step_pbar.close()
        
        block.attn.steps_method[now_stepi] = selected_method
        del loss, efficient_outputs
        
        if now_stepi == 31:
            model.total_pbar.close()
            delattr(model, 'total_pbar')
    del raw_outputs

    # 因为这只是一个transformer的一个prehook，
    # 在最终确定好所有的机制以后还会走一次transformer的forward，在那一个forward里面step会递增，因此这里需要将递增的step恢复
    for block_ in model.transformer_blocks:
        block_.attn.step = now_stepi
    
    # 在确定本次Step的计划确定之后，将Cache的开关打开，使得本次Step的Cache能够正常产生
    for block in model.transformer_blocks:
        block.attn.need_cache_output[now_stepi] = [True,True]
        block.attn.need_cache_residual[now_stepi] = [True,True]

def set_need_cahce_residual(transformer):
    for blocki, block in enumerate(transformer.transformer_blocks):
        for stepi in range(len(block.attn.need_cache_residual)-1):
            for i in range(2):
                if block.attn.steps_method[stepi+1][i] in ['full_attention','asc']:
                    block.attn.need_cache_residual[stepi][i] = False
                block.attn.need_cache_residual[-1][i] = False

def calibration(model, steps=32, threshold=0.1, window_ratio=0.125):#w

    print("Calibration for transformer!!!")
    transformer = model.transformer # model应该是cfm

    loss_thresholds = []
    for step_i in range(steps):
        sub_list = []
        for blocki in range(len(transformer.transformer_blocks)):
            threshold_i = (blocki + 1) / len(transformer.transformer_blocks) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)

    insert_wars_to_attention_forward(transformer)

    hook = transformer.register_forward_pre_hook(transformer_forward_pre_hook_for_calibration, with_kwargs=True)
    transformer.loss_thresholds = loss_thresholds
    return hook # 返回hook引用便于移除

def speedup(model,delta = None, steps=32, window_ratio=0.125):#w
    assert delta is not None
    # print("Speedup for transformer!!!")
    transformer = model.transformer # model应该是cfm
    # 加载方法
    path = f"data\\methods\\{steps}_{delta}_{window_ratio}.json"
    insert_wars_to_attention_forward(transformer, steps=steps, window_ratio=window_ratio, method_path = path)

def insert_wars_to_attention_forward(transformer, steps=32, window_ratio=0.125, method_path = None):#w
    if method_path is None:
        methods = [['full_attention', 'full_attention']] * len(transformer.transformer_blocks)
        output_shares = [[False, False]] * len(transformer.transformer_blocks)
        assert len(methods) == len(transformer.transformer_blocks)
        for block, method, output_share in zip(transformer.transformer_blocks, methods, output_shares):
            attn = block.attn
            # set some attribute
            attn.window_ratio = window_ratio
            attn.method = method
            attn.output_share = output_share
            attn.step = 0
            attn.forward = types.MethodType(efficient_attention_forward, attn)
            attn.steps_method = [['full_attention', 'full_attention']for _ in range(steps)]
            attn.need_cache_residual = [[True, True] for _ in range(steps)]
            attn.need_cache_output = [[True, True] for _ in range(steps)]
            attn.cached_residual = None
            attn.cached_output = None
    else:
        with open(method_path, 'r') as f:
            import json
            saved_methods = json.loads(open(method_path).read())['methods']

            for methods, block in zip(saved_methods, transformer.transformer_blocks):
                attn = block.attn
                attn.steps_method = methods
                attn.window_ratio = window_ratio
                attn.step = 0
                attn.forward = types.MethodType(efficient_attention_forward, attn)
                attn.need_cache_residual = [[True, True] for _ in range(steps)]
                attn.need_cache_output = [[True, True] for _ in range(steps)]
                attn.cached_residual = None
                attn.cached_output = None
                
            set_need_cahce_residual(transformer)

def efficient_attention_forward(
    self,
    x: float, 
    mask: bool | None = None, 
    rope=None,  # rotary position embedding
    block_id=None,
    enable_flash_attn=True,
): 
    batch_size = x.shape[0] // 2  # 总batch size的一半，用于分割条件和无条件部分
    outputs = []
    
    # 分别处理条件和无条件部分
    x_cond = x[:batch_size]  # 条件部分
    x_uncond = x[batch_size:]  # 无条件部分
    
    asc_index = -1
    
    for i, (curr_x, curr_method) in enumerate(zip([x_cond, x_uncond], self.steps_method[self.step])):
        # 如果是asc，直接跳过，后面复制输出
        if 'asc' in curr_method:
            asc_index = i
            continue
        # 如果是AST模式,直接使用缓存的输出
        if 'ast' in curr_method:
            if i == 0:  # 条件部分
                outputs.append(self.cached_output[:batch_size])
            else:  # 无条件部分
                outputs.append(self.cached_output[batch_size:])
            continue
            
        # 计算query, key, value投影
        query = self.to_q(curr_x)
        key = self.to_k(curr_x)
        value = self.to_v(curr_x)
        
        curr_batch_size, seq_len, _ = curr_x.shape
        
        # 应用旋转位置编码
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
            
        # 重塑attention维度
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(curr_batch_size, -1, self.heads, head_dim)
        key = key.view(curr_batch_size, -1, self.heads, head_dim)
        value = value.view(curr_batch_size, -1, self.heads, head_dim)
        
        # 计算window attention输出
        self.window_size = int(seq_len * self.window_ratio)
        w_output = flash_attn_func(query, key, value, causal=False, window_size=(-self.window_size, self.window_size))
        
        # 根据方法选择attention策略
        if 'full_attention' in curr_method:
            # 计算full attention和residual
            f_output = flash_attn_func(query, key, value, causal=False)
            w_residual = f_output - w_output
            
            device = w_output.device
            dtype = w_output.dtype
            
            # 缓存residual如果需要
            if self.need_cache_residual[self.step][i]:
                # 如果是第一次使用，初始化cached_residual
                if self.cached_residual is None:
                    self.cached_residual = torch.zeros([2,w_output.shape[1],w_output.shape[2],w_output.shape[3]],device=device,dtype=dtype)
                    
                if i == 0:  # 条件部分
                    self.cached_residual[:curr_batch_size] = w_residual
                else:  # 无条件部分
                    self.cached_residual[curr_batch_size:] = w_residual
                
            output = f_output
            
        elif 'wars' in curr_method:
            # 使用cached residual
            assert hasattr(self, 'cached_residual'), "必须要先过Full attention产生Residual output才能使用Wars"
            if i == 0:  # 条件部分
                cached_residual = self.cached_residual[:curr_batch_size]
            else:  # 无条件部分
                cached_residual = self.cached_residual[curr_batch_size:]
            output = w_output + cached_residual
        else:
            raise NotImplementedError
            
        # 重塑输出维度并应用投影
        curr_output = output.reshape(curr_batch_size, -1, self.heads * head_dim)
        curr_output = curr_output.to(query.dtype)
        curr_output = self.to_out[0](curr_output)
        curr_output = self.to_out[1](curr_output)
        
        # 应用mask如果存在
        if mask is not None:
            mask = mask.unsqueeze(-1)
            curr_output = curr_output.masked_fill(~mask, 0.0)

        # 缓存输出如果需要
        if self.need_cache_output[self.step][i]:
            # 如果是第一次使用，初始化cached_output
            if self.cached_output is None:
                self.cached_output = torch.zeros([2,curr_output.shape[1],curr_output.shape[2]],device=device,dtype=dtype)
                
            if i == 0:  # 条件部分
                self.cached_output[:curr_batch_size] = curr_output
            else:  # 无条件部分
                self.cached_output[curr_batch_size:] = curr_output
                
        outputs.append(curr_output)
        
    if asc_index >= 0:
        output_copy = outputs[0].clone()
        outputs.append(output_copy)
        # 更新缓存
        if asc_index == 0:
            if self.need_cache_output[self.step][0]:
                self.cached_output[:batch_size] = output_copy
        else:
            if self.need_cache_output[self.step][1]:
                self.cached_output[batch_size:] = output_copy
    
    # 合并条件和无条件输出
    x = torch.cat(outputs, dim=0)
    
    # 更新step
    self.step += 1
    
    return x

def save_block_output_hook(module, args, kwargs, output):
    """保存DiTBlock输出的钩子函数"""
    # 获取当前时间步
    step = module.attn.step
    
    # 获取block索引
    if not hasattr(module, 'block_id'):
        raise AttributeError("DiTBlock must have block_id attribute. Please set it during initialization.")
    block_id = module.block_id
    if block_id == 0:
        print(f'当前时间步{step}')
    # 构建保存路径
    save_dir = "data/block_outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"block_{block_id}_step_{step}.pt")
    
    # 保存输出
    torch.save(output.detach().cpu(), save_path)
    return output

def save_attn_weight_forward_pre_hook(module, args, kwargs):
    """保存Attention权重的前向钩子函数"""
    # 获取当前时间步
    step = module.step
    
    # 获取block索引
    if not hasattr(module, 'block_id'):
        raise AttributeError("DiTBlock must have block_id attribute. Please set it during initialization.")
    block_id = module.block_id
    
    # 构建保存路径
    save_dir = "attn_weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"block_{block_id}_step_{step}.pt")
    
    # 保存权重
    x = kwargs['x']
    mask = kwargs.get('mask', None)
    
    query = module.to_q(x)
    key = module.to_k(x)
    
    inner_dim = key.shape[-1]
    attn_weights = query @ key.transpose(-2,-1) / math.sqrt(inner_dim)
    if mask is not None:
        attn_weights = attn_weights.masked_fill(~mask, 0.0)
    attn_weights = F.softmax(attn_weights, dim=-1)
    torch.save(attn_weights.detach().cpu(), save_path)