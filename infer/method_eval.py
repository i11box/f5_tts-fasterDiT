from re import I
import optuna
import numpy as np
import os
import json
import time
import argparse
import sys
import torch
from copy import deepcopy
from pathlib import Path
import pickle
import optuna

from f5_tts.model.hook import (
    insert_wars_to_attention_forward,
    efficient_attention_forward,
    set_need_cahce_residual,
    compression_loss,
)
from torchdiffeq import odeint
import types
from tqdm import tqdm
import tomli

METHOD_COST_UNIT = {
    'full_attention': 0.5,
    'ast': 0,
    'asc': 0,
    'wars': 0.0625
}

METHOD_QUALITY = {}

method_candidates = [
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

def get_method_cost(method):
    assert method[1] in METHOD_COST_UNIT and method[0] in METHOD_COST_UNIT
    return METHOD_COST_UNIT[method[0]] + METHOD_COST_UNIT[method[1]]

def method2key(method):
    return method[0] + '&' + method[1]

def init_method_quality():
    global METHOD_QUALITY
    # 加载各个策略文件
    for method in method_candidates:
        key_name = method2key(method)
        METHOD_QUALITY.setdefault(key_name, {})
        with open(f"data/method_evaluation/{key_name}.json", "rb") as f:
            method_eval = json.load(f)
            for stepi in range(1,31):
                values = []
                for blocki in range(22):
                    values.append(method_eval[f'block_{blocki}'][f'step_{stepi}'])
                METHOD_QUALITY[key_name][stepi] = sum(values) / len(values)
    # 保存method_quality
    with open('data/method_quality.json', 'w', encoding='utf-8') as f:
        json.dump(METHOD_QUALITY, f, ensure_ascii=False, indent=2)
    
def get_method_quality(method,step,method_quality=None):
    if method_quality is None:
        return METHOD_QUALITY[method2key(method)][step]
    else:
        return method_quality[method2key(method)][step]

# 重排序方法候选列表
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
        score = computation_reduction * (1-step_weights[now_stepi-1]) - get_method_quality(method,now_stepi,method_quality) * step_weights[now_stepi-1]
        method_scores.append((method, score))
    
    # 按得分降序排序
    method_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 返回排序后的方法列表
    return [method for method, _ in method_scores]

# 修改后的校准函数，使用优化后的搜索顺序
def transformer_forward_pre_hook_for_eval(model, args, kwargs, step_weights):
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
            blocki
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

# 计算总策略计算量
def calculate_total_cost(transformer):
    total_cost = 0
    method_static = {
        'full_attention': 0,
        'ast': 0,
        'asc': 0,
        'wars': 0
    }
    # 统计每个块和步骤的策略计算量
    for blocki, block in enumerate(transformer.transformer_blocks):
        for stepi in range(len(block.attn.steps_method)):
            method = block.attn.steps_method[stepi]
            for i in range(2):
                if method[i] in METHOD_COST_UNIT:
                    total_cost += METHOD_COST_UNIT[method[i]] * (1.125 if block.attn.need_cache_residual[stepi][i] and method[i] == 'full_attention' else 1.0)
                    method_static[method[i]] += 1
                else:
                    raise ValueError(f"Unknown method: {method}")
    
    print(method_static)
    
    return total_cost

# 早停机制回调函数
class EarlyStoppingCallback:
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.best_value = float('inf')
        self.best_trial = None
        self.early_stop = False
        
    def __call__(self, study, trial):
        if study.best_value < self.best_value - self.threshold:
            # 有显著改善
            self.best_value = study.best_value
            self.best_trial = study.best_trial
        elif trial.number >= 5:  # 至少运行5次试验后才考虑早停
            # 检查最近几次试验是否有改善
            recent_values = [t.value for t in study.trials[-5:] if t.value is not None]
            if recent_values and min(recent_values) > self.best_value - self.threshold:
                # 最近5次试验没有显著改善
                self.early_stop = True
                return True
        return False

# 定义目标函数
def objective(trial, model_obj, nfe_step, threshold):
    # 每个时间步的权重作为超参数
    step_weights = []
    for i in range(1, 13):  # 1-12步，第0步不使用
        # 不重要的权重
        if i in [5,6,9,10,11,12]:
            step_weights.append(0.0)
        else:
            step_weights.append(trial.suggest_float(f'step_weight_{i}', 0.0, 0.5))
    
    # 设置损失阈值
    loss_thresholds = []
    transformer = model_obj
    for step_i in range(nfe_step):
        sub_list = []
        for blocki in range(len(transformer.transformer_blocks)):
            threshold_i = (blocki + 1) / len(transformer.transformer_blocks) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)
    
    transformer.loss_thresholds = loss_thresholds
    
    # 初始化模型设置
    insert_wars_to_attention_forward(transformer, steps=nfe_step)
    
    # 注册钩子函数
    hook = lambda *args, **kwargs: transformer_forward_pre_hook_for_eval(*args, **kwargs, step_weights=step_weights)
    forward_hook = transformer.register_forward_pre_hook(hook, with_kwargs=True)
    
    try:
        # 检查是否有可用的GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移到设备上并转为半精度
        transformer = transformer.to(device=device,dtype=torch.float16)
        
        # 加载保存的y0和t数据并转为半精度
        y0 = torch.load("data/y0_t.pt").to(device=device,dtype=torch.float16)
        t = torch.load("data/t_inter.pt").to(device=device,dtype=torch.float16)  
        text = torch.load("data/text.pt").to(device=device,dtype=torch.int64)
        step_cond = torch.load("data/step_cond.pt").to(device=device,dtype=torch.float16)
        
        # 定义前向函数
        def fn(t, x, step_cond, text):
            # 确保数据类型一致
            if isinstance(t, float):
                t = torch.tensor([t], device=device)
            
            x, step_cond, text = x.repeat(2, 1, 1), step_cond.repeat(2, 1, 1), text.repeat(2, 1)
            

            pred = transformer(
                x=x, cond=step_cond, text=text, time=t, mask=None, drop_audio_cond=True, drop_text=True
            )
            pred, null_pred = pred.chunk(2)
            
            # 应用CFG
            return pred + (pred - null_pred)
        
        # 运行odeint
        with torch.inference_mode():
            # 设置较小的批处理大小以避免内存不足
            batch_size = y0.shape[0]
            _ = odeint(
                lambda t, x: fn(t, x, step_cond, text),
                y0,
                t,
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        
        # 设置need_cache_residual
        set_need_cahce_residual(transformer)
        if not hasattr(model_obj, 'trial_progress'):
            model_obj.trial_progress = tqdm(total=20, desc="试验进度", position=1)
        else:
            model_obj.trial_progress.update(1)
        # 计算总策略计算量
        total_cost = calculate_total_cost(transformer)
        # 我们的目标是减少计算量，所以返回负值（optuna默认最小化目标函数）
        return total_cost
        
    finally:
        # 清理钩子函数
        if forward_hook is not None:
            forward_hook.remove()

def save_best(model_obj, nfe_step, threshold, params_path='data/optuna_result/best_params.json', output_path='data/best_strategy.json'):
    """
    读取best_params.json，使用最优参数进行校准并保存最优策略组合
    
    Args:
        model_obj: 模型对象
        nfe_step: 步数
        threshold: 阈值
        params_path: best_params.json路径
        output_path: 输出策略文件路径
    """
    import logging
    logger = logging.getLogger("method_eval")
    logger.info("开始保存最优策略...")
    
    # 1. 加载最优参数
    logger.info(f"从{params_path}加载最优参数")
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)
    except Exception as e:
        logger.error(f"无法加载最优参数: {e}")
        return
    
    # 2. 提取参数
    # 提取step_weights
    step_weights = []
    for i in range(1, 13):  # 1-12步，第0步不使用
        if i in [5, 6, 9, 10, 11, 12]:
            step_weights.append(0.0)
        else:
            param_name = f'step_weight_{i}'
            if param_name in best_params:
                step_weights.append(best_params[param_name])
            else:
                step_weights.append(0.0)
    
    logger.info(f"参数提取完成:")
    logger.info(f"- step_weights: {step_weights}")
    
    # 设置损失阈值
    loss_thresholds = []
    transformer = model_obj
    for step_i in range(nfe_step):
        sub_list = []
        for blocki in range(len(transformer.transformer_blocks)):
            threshold_i = (blocki + 1) / len(transformer.transformer_blocks) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)
    
    transformer.loss_thresholds = loss_thresholds
    
    # 初始化模型设置
    from f5_tts.model.hook import insert_wars_to_attention_forward
    insert_wars_to_attention_forward(transformer, steps=nfe_step, window_ratio=0.125)
    
    # 注册钩子函数
    hook = lambda *args, **kwargs: transformer_forward_pre_hook_for_eval(*args, **kwargs, step_weights=step_weights)
    forward_hook = transformer.register_forward_pre_hook(hook, with_kwargs=True)
    
    try:
        # 检查是否有可用的GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移到设备上并转为半精度
        transformer = transformer.to(device=device,dtype=torch.float16)
        
        # 加载保存的y0和t数据并转为半精度
        y0 = torch.load("data/y0_t.pt").to(device=device,dtype=torch.float16)
        t = torch.load("data/t_inter.pt").to(device=device,dtype=torch.float16)  
        text = torch.load("data/text.pt").to(device=device,dtype=torch.int64)
        step_cond = torch.load("data/step_cond.pt").to(device=device,dtype=torch.float16)
        
        # 定义前向函数
        def fn(t, x, step_cond, text):
            # 确保数据类型一致
            if isinstance(t, float):
                t = torch.tensor([t], device=device)
            
            x, step_cond, text = x.repeat(2, 1, 1), step_cond.repeat(2, 1, 1), text.repeat(2, 1)
            
            pred = transformer(
                x=x, cond=step_cond, text=text, time=t, mask=None, drop_audio_cond=True, drop_text=True
            )
            pred, null_pred = pred.chunk(2)
            
            # 应用CFG
            return pred + (pred - null_pred)
        
        # 运行odeint
        with torch.inference_mode():
            # 设置较小的批处理大小以避免内存不足
            batch_size = y0.shape[0]
            trajectory = []
            
            # 根据GPU内存情况调整批处理大小
            if torch.cuda.is_available():
                try:
                    # 尝试一次性计算
                    from torchdiffeq import odeint
                    trajectory = odeint(
                        lambda t, x: fn(t, x, step_cond, text),
                        y0,
                        t,
                        atol=1e-4,
                        rtol=1e-4,
                        method="euler",
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("GPU内存不足，尝试分批计算...")
                        # 释放内存
                        torch.cuda.empty_cache()
                        
                        # 分批计算
                        chunk_size = batch_size // 2  # 或更小的值
                        for i in range(0, batch_size, chunk_size):
                            end = min(i + chunk_size, batch_size)
                            y0_chunk = y0[i:end]
                            step_cond_chunk = step_cond[i:end]
                            text_chunk = text[i:end]
                            
                            chunk_trajectory = odeint(
                                lambda t, x: fn(t, x, step_cond_chunk, text_chunk),
                                y0_chunk,
                                t,
                                atol=1e-4,
                                rtol=1e-4,
                                method="euler",
                            )
                            
                            if not trajectory:
                                trajectory = chunk_trajectory
                            else:
                                # 合并结果
                                trajectory = torch.cat([trajectory, chunk_trajectory], dim=1)
                            
                            # 释放内存
                            del chunk_trajectory
                            torch.cuda.empty_cache()
                    else:
                        raise e
            else:
                # CPU计算
                from torchdiffeq import odeint
                trajectory = odeint(
                    lambda t, x: fn(t, x, step_cond, text),
                    y0,
                    t,
                    atol=1e-4,
                    rtol=1e-4,
                    method="euler",
                )
        
        # 设置need_cache_residual
        from f5_tts.model.hook import set_need_cahce_residual
        set_need_cahce_residual(transformer)
        
        # 收集并保存策略
        logger.info("校准完成，收集策略...")
        saved_methods = []
        for block in transformer.transformer_blocks:
            block_methods = []
            for stepi in range(nfe_step):
                # 获取当前步骤的最优方法
                if hasattr(block.attn, 'steps_method') and stepi < len(block.attn.steps_method):
                    method = block.attn.steps_method[stepi]
                    block_methods.append(method)
                else:
                    # 默认方法
                    block_methods.append(['full_attention', 'full_attention'])
            saved_methods.append(block_methods)
        
        # 保存策略到JSON文件
        strategy_data = {
            "methods": saved_methods
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(strategy_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"最优策略已保存到: {output_path}")
        
    finally:
        # 清理钩子函数
        if forward_hook is not None:
            forward_hook.remove()

def save_best_helper():
    import logging
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/method_eval.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("method_eval")
    
    try:
        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/method_evaluation", exist_ok=True)
        os.makedirs("data/optuna_result", exist_ok=True)
        
        logger.info("开始加载模型...")
        # 加载模型
        from f5_tts.model.backbones.dit import DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        model_obj = DiT(**model_cfg,text_num_embeds=2545,mel_dim=100)
        
        # 检查是否有可用的GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        model_obj = model_obj.to(device)
        model_obj.eval()
        logger.info("模型加载完成")
        
        # 检查必要文件是否存在
        required_files = ["data/y0_t.pt", "data/t_inter.pt", "data/text.pt", "data/step_cond.pt"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.error(f"缺少必要文件: {missing_files}")
            logger.error("请确保已经运行过模型并保存了必要的张量数据")
            return
        
        # 创建output_dir目录
        output_dir = f'data/optuna_result'
        os.makedirs(output_dir, exist_ok=True)
        
        steps = 32
        threshold = 0.2
        init_method_quality() # 初始化method_quality

        save_best(model_obj, steps, threshold)
            
    except Exception as e:
        logger.exception(f"执行过程中发生错误: {e}")
        raise

def main():
    import logging
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/method_eval.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("method_eval")
    
    try:
        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/method_evaluation", exist_ok=True)
        os.makedirs("data/optuna_result", exist_ok=True)
        
        logger.info("开始加载模型...")
        # 加载模型
        from f5_tts.model.backbones.dit import DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        model_obj = DiT(**model_cfg,text_num_embeds=2545,mel_dim=100)
        
        # 检查是否有可用的GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        model_obj = model_obj.to(device)
        model_obj.eval()
        logger.info("模型加载完成")
        
        # 检查必要文件是否存在
        required_files = ["data/y0_t.pt", "data/t_inter.pt", "data/text.pt", "data/step_cond.pt"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.error(f"缺少必要文件: {missing_files}")
            logger.error("请确保已经运行过模型并保存了必要的张量数据")
            return
        
        # 创建output_dir目录
        output_dir = f'data/optuna_result'
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查是否有之前的优化结果
        study_save_path = os.path.join(output_dir, 'study.pkl')
        if os.path.exists(study_save_path):
            logger.info(f"发现之前的优化结果，从断点继续优化: {study_save_path}")
            with open(study_save_path, 'rb') as f:
                study = pickle.load(f)
            logger.info(f"已完成的试验数: {len(study.trials)}")
            logger.info(f"当前最佳值: {study.best_value}")
        else:
            # 创建新的study对象
            logger.info("创建新的优化研究...")
            study = optuna.create_study(direction='minimize')
        
        steps = 32
        threshold = 0.2
        init_method_quality() # 初始化method_quality
        # 创建早停回调
        early_stopping_callback = EarlyStoppingCallback(threshold=0.05)
        
        # 运行优化
        n_trials = 20
        
        logger.info(f"开始优化，共{steps}步，threshold={threshold}")
        try:
            study.optimize(
                lambda trial: objective(
                    trial, 
                    model_obj, 
                    steps, 
                    threshold
                ), 
                n_trials=n_trials,  # 运行20次优化
                catch=(Exception,),
                callbacks=[early_stopping_callback]
            )
        except optuna.exceptions.TrialPruned:
            logger.info("优化已被早停机制终止")
        
        # 保存优化study以便下次继续
        with open(study_save_path, 'wb') as f:
            pickle.dump(study, f)
        logger.info(f"优化study已保存至 {study_save_path}")
        
        # 获取最佳参数
        best_params = study.best_params
        best_value = study.best_value
        
        # 保存结果
        result = {
            'best_params': best_params,
            'best_value': best_value,
            'completed_trials': len(study.trials),
            'early_stopped': early_stopping_callback.early_stop
        }
        
        result_path = os.path.join(output_dir, 'best_params.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"优化完成，最佳参数已保存至 {result_path}")
        logger.info(f"最佳计算量减少: {-best_value}")
        logger.info(f"完成试验数: {len(study.trials)}")
        if early_stopping_callback.early_stop:
            logger.info("优化已提前结束，因为连续多次试验没有显著改善")
        
        # 可视化结果
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(os.path.join(output_dir, 'optimization_history.html'))
            
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(os.path.join(output_dir, 'param_importances.html'))
            
            fig = optuna.visualization.plot_intermediate_values(study)
            fig.write_html(os.path.join(output_dir, 'intermediate_values.html'))
            
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_html(os.path.join(output_dir, 'parallel_coordinate.html'))
            
            # 绘制轮廓图 (仅适用于两个参数的情况)
            if len(study.best_params) == 2:
                fig = optuna.visualization.plot_contour(study, params=list(study.best_params.keys()))
                fig.write_html(os.path.join(output_dir, 'contour.html'))
            
            logger.info(f"可视化结果已保存至 {output_dir}")
        except Exception as e:
            logger.warning(f"无法创建可视化结果: {e}")
            
    except Exception as e:
        logger.exception(f"执行过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    save_best_helper()
