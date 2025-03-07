from copy import deepcopy
import glob
import json
from multiprocessing import Process, Queue
import os
import types
from typing import Any, Optional, Tuple
from skimage.filters import threshold_otsu
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

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """计算两个图像的结构相似性指数
    
    Args:
        img1: 第一个图像张量
        img2: 第二个图像张量
        window_size: 高斯窗口大小
        size_average: 是否对结果取平均
        
    Returns:
        ssim值
    """
    # 确保输入在同一设备上并且是相同的数据类型
    device = img1.device
    dtype = img1.dtype
    
    # 将输入转换为float32以进行精确计算
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    
    # 创建高斯窗口
    def gaussian(window_size, sigma=1.5):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    # 创建1D高斯窗口
    _1D_window = gaussian(window_size).to(device)
    # 创建2D高斯窗口
    _2D_window = _1D_window.unsqueeze(1).matmul(_1D_window.unsqueeze(0))
    window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
    
    # 计算SSIM
    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 将结果转换回原始数据类型
    result = ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)
    return result

def method2key(method):
    return method[0] + '&' + method[1]

def get_method_quality(method,step,method_quality=None):
    if method_quality is None:
        raise ValueError('should have method quality')
    else:
        return method_quality[method2key(method)][str(step)]

def get_method_cost(method):
    METHOD_COST_UNIT = {
        'full_attention': 0.5,
        'ast': 0,
        'asc': 0,
        'wars': 0.0625
    }
    assert method[1] in METHOD_COST_UNIT and method[0] in METHOD_COST_UNIT
    return METHOD_COST_UNIT[method[0]] + METHOD_COST_UNIT[method[1]]

def reorder_method_candidates(method_candidates, step_weights, now_stepi, blocki=None,method_quality =None):
    """
    根据时间步权重对方法候选列表进行重新排序
    
    Args:
        method_candidates: 原始方法候选列表
        step_weights: 时间步权重向量
        now_stepi: 当前时间步
        blocki: 可选，当前块号
    
    Returns:
        重排序后的方法候选列表
    """
    # 为每个方法计算得分
    # 得分 = 计算量减少 * 时间步权重
    if now_stepi == 0:  # 第0步使用默认排序
        return method_candidates
    
    if now_stepi >= 12:
        now_stepi = 12 # 12步后都按12步的排序来
        
    method_scores = []
    for method in method_candidates:
        # 计算量减少 = 1.0 - METHOD_COST[method_key]
        computation_reduction = 1.0 - get_method_cost(method)
        # 乘以当前时间步的权重
        assert method_quality is not None
        score = computation_reduction * (1-step_weights[now_stepi-1]) - get_method_quality(method,now_stepi,method_quality) * step_weights[now_stepi-1]
        method_scores.append((method, score))
    
    # 按得分降序排序
    method_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 返回排序后的方法列表
    return [method for method, _ in method_scores]

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


def calculate_ff_flops_hook(module, args, kwargs):
    # 从kwargs中获取hidden_states
    hidden_states = args[0]
    batch_size, seq_len, dim = hidden_states.shape
    project_in = module.ff[0]
    first_linear = project_in[0]  # Sequential中的第一个Linear
    inner_dim = first_linear.out_features
    
    # 基础计算量：
    # 第一个Linear: dim -> inner_dim
    # 第二个Linear: inner_dim -> dim
    base_ops = (
        batch_size * seq_len * dim * inner_dim +  # 第一个Linear
        batch_size * seq_len * inner_dim * dim    # 第二个Linear
    )
    
    # 记录全精度计算量
    module.full_ops += base_ops
    
    # 获取当前方法
    method = module.steps_method[module.step]

    cond_op = deepcopy(base_ops)
    uncond_op = deepcopy(base_ops)
    op = [cond_op /2, uncond_op /2]
    
    # 根据不同方法计算实际计算量
    for i in range(2):
        if method[i] == "ast" or method[i] == 'asc':
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

def pre_calibration_hook(module, args, kwargs):
    """预校准，通过比较模型各层各时间步热力图与对角线的差距确定模型贪心搜索模式"""
    # 获取当前时间步
    step = module.step
    
    # 获取block索引
    if not hasattr(module, 'block_id'):
        raise AttributeError("DiTBlock must have block_id attribute. Please set it during initialization.")
    block_id = module.block_id
    
    # 保存权重
    x = kwargs['x']
    x,_ = x.chunk(2,dim=0) # 只要条件的
    mask = kwargs.get('mask', None)
    
    query = module.to_q(x).to(dtype = torch.bfloat16)
    key = module.to_k(x).to(dtype = torch.bfloat16)
    
    inner_dim = key.shape[-1]
    attn_weights = query @ key.transpose(-2,-1) / math.sqrt(inner_dim) # 获取注意力权重
    if mask is not None:
        attn_weights = attn_weights.masked_fill(~mask, 0)
    attn_weights = F.softmax(attn_weights, dim=-1)
    _,n,_ = attn_weights.shape
    diagonal_matrix = torch.eye(n, device=attn_weights.device,dtype=attn_weights.dtype)
    #------------------直接余弦版------------------------
    # 计算余弦相似度
    batch_size = attn_weights.shape[0]
    similarities = []

    for b in range(batch_size):
        # 获取当前批次的注意力权重
        attn_mat = attn_weights[b]
        
        # 将矩阵展平为向量
        attn_vec = attn_mat.reshape(-1)
        diag_vec = diagonal_matrix.reshape(-1)
        
        # 计算余弦相似度
        # 归一化向量
        attn_norm = attn_vec / torch.norm(attn_vec)
        diag_norm = diag_vec / torch.norm(diag_vec)
        
        # 计算点积
        sim = torch.dot(attn_norm, diag_norm)
        similarities.append(sim.item())
        
    # 取平均值作为最终相似度
    similarity = sum(similarities) / len(similarities)
    #------------------直接余弦版------------------------
    
    #---------------------直接计算SSIM版------------------------
    # # 计算SSIM
    # batch_size = attn_weights.shape[0]
    # similarities = []

    # for b in range(batch_size):
    #     # 获取当前批次的注意力权重
    #     attn_mat = attn_weights[b]
        
    #     # 准备输入格式：[B, C, H, W]
    #     # 注意力权重和对角矩阵都需要扩展维度
    #     attn_input = attn_mat.unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]
    #     diag_input = diagonal_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]
        
    #     # 计算SSIM
    #     # data_range是可能的最大值差异，对于softmax后的注意力权重，通常为1
    #     sim = calculate_ssim(attn_input, diag_input, window_size=11)
    #     similarities.append(sim.item())
        
    # # 取平均值作为最终相似度
    # similarity = sum(similarities) / len(similarities)
    #---------------------直接计算SSIM版------------------------
    
    
    
    #--------------特征计算然后余弦相似度---------------------
    # # 计算特征
    # batch_size = attn_weights.shape[0]
    # features_list = []
    
    # for b in range(batch_size):
    #     # 将注意力权重矩阵展平为特征向量
    #     attn_flat = attn_weights[b].flatten()
        
    #     # 计算统计特征
    #     std_dev = torch.std(attn_weights[b])
        
    #     # 计算梯度特征 (模拟numpy的gradient)
    #     grad_y = attn_weights[b][:, 1:] - attn_weights[b][:, :-1]
    #     grad_x = attn_weights[b][1:, :] - attn_weights[b][:-1, :]
    #     # 填充使尺寸一致
    #     grad_y = F.pad(grad_y, (0, 1), "constant", 0)
    #     grad_x = F.pad(grad_x, (0, 0, 0, 1), "constant", 0)
        
    #     # 计算梯度幅度
    #     gradient_magnitude = torch.sqrt(grad_y**2 + grad_x**2 + 1e-10)
    #     mean_gradient = torch.mean(gradient_magnitude)
        
    #     # 计算与对角线的相似度
    #     diag_similarity = torch.sum(attn_weights[b] * diagonal_matrix) / (torch.sum(diagonal_matrix) + 1e-10)
        
    #     # 组合特征
    #     combined_features = torch.cat([
    #         attn_flat,
    #         torch.tensor([diag_similarity, std_dev, mean_gradient], device=attn_weights.device)
    #     ])
        
    #     features_list.append(combined_features)
    
    # # 合并批次特征
    # all_features = torch.stack(features_list)
    
    # # 2. 计算与对角线模板的相似度
    # # 创建对角线模板特征
    # diagonal_template = diagonal_matrix.flatten()
    # template_std = torch.std(diagonal_matrix)
    # # 对角线矩阵的梯度
    # diag_grad_y = diagonal_matrix[:, 1:] - diagonal_matrix[:, :-1]
    # diag_grad_x = diagonal_matrix[1:, :] - diagonal_matrix[:-1, :]
    # diag_grad_y = F.pad(diag_grad_y, (0, 1), "constant", 0)
    # diag_grad_x = F.pad(diag_grad_x, (0, 0, 0, 1), "constant", 0)
    # diag_gradient_magnitude = torch.sqrt(diag_grad_y**2 + diag_grad_x**2 + 1e-10)
    # diag_mean_gradient = torch.mean(diag_gradient_magnitude)
    
    # # 构建模板特征
    # diag_diag_similarity = 1.0  # 对角线与自身的相似度为1
    # template_features = torch.cat([
    #     diagonal_template,
    #     torch.tensor([diag_diag_similarity, template_std, diag_mean_gradient], device=attn_weights.device)
    # ])
    
    # # 计算余弦相似度
    # template_features = template_features.unsqueeze(0)  # [1, feature_dim]
    
    # # 归一化特征向量
    # template_norm = F.normalize(template_features, p=2, dim=1)
    # features_norm = F.normalize(all_features, p=2, dim=1)
    
    # # 计算余弦相似度
    # similarity = torch.sum(template_norm * features_norm, dim=1)
    #--------------特征计算然后余弦相似度---------------------
    
    # 记录相似度
    if not hasattr(module, 'diagonal_similarities'):
        module.diagonal_similarities = {}
    module.diagonal_similarities[step] = similarity
    
    print(f"Block {block_id}, Step {step}, 与对角线相似度: {similarity:.4f}")
    
    module.step += 1


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
        block.ff.need_cache_output[now_stepi] = [False, False]

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
            # 修改ff的方法
            block.ff.steps_method[now_stepi] = method

            for block_ in model.transformer_blocks:
                block_.attn.step = now_stepi
                block_.ff.step = now_stepi
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
        block.ff.steps_method[now_stepi] = selected_method
        del loss, efficient_outputs
        
        if now_stepi == 31:
            model.total_pbar.close()
            delattr(model, 'total_pbar')
    del raw_outputs

    # 因为这只是一个transformer的一个prehook，
    # 在最终确定好所有的机制以后还会走一次transformer的forward，在那一个forward里面step会递增，因此这里需要将递增的step恢复
    for block_ in model.transformer_blocks:
        block_.attn.step = now_stepi
        block_.ff.step = now_stepi

    # 在确定本次Step的计划确定之后，将Cache的开关打开，使得本次Step的Cache能够正常产生
    for block in model.transformer_blocks:
        block.attn.need_cache_output[now_stepi] = [True,True]
        block.attn.need_cache_residual[now_stepi] = [True,True]
        block.ff.need_cache_output[now_stepi] = [True, True]

# 修改后的校准函数，使用优化后的搜索顺序
def transformer_forward_pre_hook_for_eval(model, args, kwargs, step_weights):
    method_quality = json.load(open('data/method_quality.json', 'r', encoding='utf-8'))
    now_stepi = model.transformer_blocks[0].attn.step

    if not hasattr(model, 'calibration_progress'):
        model.calibration_progress = tqdm(total=32, desc="校准进度", position=0)
        model.calibration_progress.update(now_stepi)
    else:
        model.calibration_progress.update(1)

    # 为了避免在搜索candidate method时对cache内容产生改变，因此搜索时需要先关掉cache的开关
    for block in model.transformer_blocks:
        block.attn.forward = types.MethodType(efficient_attention_forward, block.attn)
        block.attn.need_cache_output[now_stepi] = [False, False]
        block.attn.need_cache_residual[now_stepi] = [False, False]

    # 原始方法候选列表
    default_method_candidates = [
        ['ast', 'ast'],
        ['ast', 'asc'],
        ['asc', 'ast'],
        ['ast', 'wars'],
        ['wars', 'ast'],
        ['wars', 'asc'],
        ['asc', 'wars'],
        ['wars', 'wars'],
        ['full_attention', 'ast'],
        ['ast', 'full_attention'],
        ['full_attention', 'asc'],
        ['asc', 'full_attention'],
        ['wars', 'full_attention'],
        ['full_attention', 'wars'],
    ]
    
    # 先走一遍得到full-attention的值
    raw_outputs = model.forward(*args, **kwargs)
    raw_output_cond, raw_output_uncond = raw_outputs.chunk(2, dim=0)
    raw_outputs = 2 * raw_output_cond - raw_output_uncond
    
    for blocki, block in enumerate(model.transformer_blocks):
        if now_stepi == 0:
            continue
            
        # 使用加权排序的方法列表
        method_candidates = reorder_method_candidates(
            deepcopy(default_method_candidates), 
            step_weights, 
            now_stepi, 
            blocki,
            method_quality
        )
        
        # method的由强到弱
        selected_method = ['full_attention', 'full_attention']  # 第一个代表cond
        for method in method_candidates:
            block.attn.steps_method[now_stepi] = method

            for block_ in model.transformer_blocks:
                block_.attn.step = now_stepi
            efficient_outputs = model.forward(*args, **kwargs)
            efficient_output_cond, efficient_output_uncond = efficient_outputs.chunk(2, dim=0)
            efficient_outputs = 2 * efficient_output_cond - efficient_output_uncond
            loss = compression_loss(raw_outputs, efficient_outputs)
            threshold = model.loss_thresholds[now_stepi][blocki]

            if loss < threshold:
                selected_method = method
                break
                    
        block.attn.steps_method[now_stepi] = selected_method
        del loss, efficient_outputs
    
    del raw_outputs

    # 因为这只是一个transformer的一个prehook，
    # 在最终确定好所有的机制以后还会走一次transformer的forward，在那一个forward里面step会递增，因此这里需要将递增的step恢复
    for block_ in model.transformer_blocks:
        block_.attn.step = now_stepi
    
    # 在确定本次Step的计划确定之后，将Cache的开关打开，使得本次Step的Cache能够正常产生
    for block in model.transformer_blocks:
        block.attn.need_cache_output[now_stepi] = [True, True]
        block.attn.need_cache_residual[now_stepi] = [True, True]

def set_need_cahce_residual(transformer):
    for blocki, block in enumerate(transformer.transformer_blocks):
        for stepi in range(len(block.attn.need_cache_residual)-1):
            for i in range(2):
                if block.attn.steps_method[stepi+1][i] in ['full_attention','asc']:
                    block.attn.need_cache_residual[stepi][i] = False
                block.attn.need_cache_residual[-1][i] = False

def pre_calibration(model):
    print("Pre Calibration for transformer!!!")
    transformer = model.transformer # model应该是cfm
    # 关掉缓存
    for block in transformer.transformer_blocks:
        block.attn.need_cache_output = [False,False]
        block.attn.need_cache_residual = [False,False]
        block.ff.need_cache_output = [False, False]
    hooks = []
    for blocki in range(len(transformer.transformer_blocks)):
        block = transformer.transformer_blocks[blocki]
        hooks.append(block.attn.register_forward_pre_hook(pre_calibration_hook, with_kwargs=True))
    return hooks

def calibration(model, steps=32, threshold=0.1, window_ratio=0.125,is_eval = False):

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

def eval_method(model, step_weights, steps=32, threshold=0.2,):
    
    print("Eval Method for transformer!!!")
    transformer = model.transformer # model应该是cfm

    loss_thresholds = []
    for step_i in range(steps):
        sub_list = []
        for blocki in range(len(transformer.transformer_blocks)):
            threshold_i = (blocki + 1) / len(transformer.transformer_blocks) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)

    insert_wars_to_attention_forward(transformer)
    
    fn_eval = lambda *args, **kwargs: transformer_forward_pre_hook_for_eval(*args, **kwargs, step_weights=step_weights)
    
    hook = transformer.register_forward_pre_hook(fn_eval, with_kwargs=True)

    transformer.loss_thresholds = loss_thresholds
    return hook # 返回hook引用便于移除

def speedup(model,delta = None, steps=32, window_ratio=0.125):
    assert delta is not None
    # print("Speedup for transformer!!!")
    transformer = model.transformer # model应该是cfm
    # 加载方法
    path = f"data\\methods\\{steps}_{delta}_{window_ratio}.json"
    insert_wars_to_attention_forward(transformer, steps=steps, window_ratio=window_ratio, method_path = path)

def insert_wars_to_attention_forward(transformer, steps=32, window_ratio=0.125, method_path = None):
    if method_path is None:
        methods = [['full_attention', 'full_attention']] * len(transformer.transformer_blocks)
        output_shares = [[False, False]] * len(transformer.transformer_blocks)
        assert len(methods) == len(transformer.transformer_blocks)
        for block, method, output_share in zip(transformer.transformer_blocks, methods, output_shares):
            attn = block.attn
            ff = block.ff
            # for attn set some attribute
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
            # for ff set some attribute
            ff.method = method
            ff.steps_method = [['full_attention', 'full_attention']for _ in range(steps)]
            ff.need_cache_output = [[True, True] for _ in range(steps)]
            ff.output_share = output_share
            ff.step = 0
            ff.forward = types.MethodType(efficient_ff_forward, ff)
            ff.cached_output = None
    else:
        with open(method_path, 'r') as f:
            import json
            saved_methods = json.loads(open(method_path).read())['methods']

            for methods, block in zip(saved_methods, transformer.transformer_blocks):
                # for attn
                attn = block.attn
                attn.steps_method = methods
                attn.window_ratio = window_ratio
                attn.step = 0
                attn.forward = types.MethodType(efficient_attention_forward, attn)
                attn.need_cache_residual = [[True, True] for _ in range(steps)]
                attn.need_cache_output = [[True, True] for _ in range(steps)]
                attn.cached_residual = None
                attn.cached_output = None
                # for ff
                ff = block.ff
                ff.steps_method = methods
                ff.need_cache_output = [[True, True] for _ in range(steps)]
                ff.output_share = False
                ff.step = 0
                ff.forward = types.MethodType(efficient_ff_forward, ff)
                ff.cached_output = None

            set_need_cahce_residual(transformer)

def efficient_ff_forward(self, x):
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
            
        # 计算前馈网络输出
        curr_output = self.ff(curr_x)
        
        # 缓存输出如果需要
        if self.need_cache_output[self.step][i]:
            # 如果是第一次使用，初始化cached_output
            if self.cached_output is None:
                device = curr_output.device
                dtype = curr_output.dtype
                self.cached_output = torch.zeros([2, curr_output.shape[1], curr_output.shape[2]], device=device, dtype=dtype)
                
            if i == 0:  # 条件部分
                self.cached_output[:batch_size] = curr_output
            else:  # 无条件部分
                self.cached_output[batch_size:] = curr_output
                
        outputs.append(curr_output)
        
    if asc_index >= 0:
        output_copy = outputs[0].clone()
        outputs.insert(asc_index, output_copy)
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
    
    # # 构建保存路径
    # save_dir = "attn_weights"
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"block_{block_id}_step_{step}.pt")
    
    # 保存权重
    x = kwargs['x']
    mask = kwargs.get('mask', None)
    
    query = module.to_q(x).to(dtype = torch.bfloat16)
    key = module.to_k(x).to(dtype = torch.bfloat16)
    
    inner_dim = key.shape[-1]
    attn_weights = query @ key.transpose(-2,-1) / math.sqrt(inner_dim)

    # torch.save(query.detach().cpu(), f'step{step}_query.pt')
    # torch.save(key.detach().cpu(), f'step{step}_key.pt')
    # print(f'inner_dim: {inner_dim} step{step}')
    # torch.save(attn_weights.detach().cpu(), f'step{step}_attn_weights_before_softmax.pt')
    if mask is not None:
        # print(f'using mask in step{step}')
        attn_weights = attn_weights.masked_fill(~mask, 0)
    attn_weights = F.softmax(attn_weights, dim=-1)
    if torch.isnan(attn_weights).any():
        print(f"NaN detected in attn_weights after softmax for block_{block_id}_step_{step}")
    # torch.save(attn_weights.detach().cpu(), f'step{step}_attn_weights_after_softmax.pt')
    # torch.save(attn_weights.detach().cpu(), save_path)