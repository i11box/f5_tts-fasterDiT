import os
import torch
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from collections import defaultdict

def load_predictions(old_dir, new_dir):
    """加载所有时间步的预测结果"""
    predictions = defaultdict(dict)
    
    # 加载新版本预测
    for file in glob(os.path.join(new_dir, "*.pt")):
        data = torch.load(file)
        time_step = data['time_step']
        predictions[time_step]['new'] = {
            'output': data['output']
        }
    
    # 加载旧版本预测
    for file in glob(os.path.join(old_dir, "*.pt")):
        data = torch.load(file)
        time_step = data['time_step']
        predictions[time_step]['old'] = {
            'output': data['output']
        }
    
    return predictions

def compute_statistics(tensor):
    """计算张量的基本统计信息"""
    return {
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item()
    }

def compare_outputs(old_dir="./out/data/old", new_dir="./out/data/new"):
    predictions = load_predictions(old_dir, new_dir)
    time_steps = sorted(predictions.keys())
    
    # 准备画图数据
    times = []
    old_means = []
    new_means = []
    old_stds = []
    new_stds = []
    
    print("\nDetailed comparison at each time step:")
    print("-" * 80)
    
    for t in time_steps:
        old_stats = compute_statistics(predictions[t]['old']['output'])
        new_stats = compute_statistics(predictions[t]['new']['output'])
        
        # 收集数据用于画图
        times.append(t)
        old_means.append(old_stats['mean'])
        new_means.append(new_stats['mean'])
        old_stds.append(old_stats['std'])
        new_stds.append(new_stats['std'])
        
        # 计算输出差异
        output_diff = (predictions[t]['old']['output'] - predictions[t]['new']['output']).abs()
        diff_stats = compute_statistics(output_diff)
        
        print(f"\nTime step: {t:.3f}")
        print(f"Old version - mean: {old_stats['mean']:.4f}, std: {old_stats['std']:.4f}")
        print(f"New version - mean: {new_stats['mean']:.4f}, std: {new_stats['std']:.4f}")
        print(f"Absolute difference - mean: {diff_stats['mean']:.4f}, max: {diff_stats['max']:.4f}")
    
    # 绘制统计图
    plt.figure(figsize=(12, 8))
    
    # 绘制均值对比
    plt.subplot(2, 1, 1)
    plt.plot(times, old_means, label='Old Version', marker='o')
    plt.plot(times, new_means, label='New Version', marker='x')
    plt.title('Mean Output Values Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True)
    
    # 绘制标准差对比
    plt.subplot(2, 1, 2)
    plt.plot(times, old_stds, label='Old Version', marker='o')
    plt.plot(times, new_stds, label='New Version', marker='x')
    plt.title('Output Standard Deviation Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output_comparison.png')
    print("\nComparison plot saved as 'output_comparison.png'")

if __name__ == "__main__":
    compare_outputs()
