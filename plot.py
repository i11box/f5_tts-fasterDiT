import os
import torch
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

def load_predictions(data_dir):
    """加载所有时间步的预测结果"""
    predictions = {}
    
    for file in sorted(glob(os.path.join(data_dir, "*.pt"))):
        data = torch.load(file)
        time_step = data['time_step']
        predictions[time_step] = data['output']
    
    return predictions

def create_heatmap(predictions, save_path='heatmap.png'):
    # 获取所有时间步
    time_steps = sorted(predictions.keys())
    
    # 准备数据
    data = []
    for t in time_steps:
        # 取第一个样本的数据，压缩最后一个维度（取平均）
        sample_data = predictions[t][0].mean(dim=-1).cpu().numpy()
        data.append(sample_data)
    
    # 转换为numpy数组
    data = np.array(data)
    
    # 创建热力图
    plt.figure(figsize=(15, 10))
    
    # 使用seaborn的热力图，保存返回值
    heatmap = sns.heatmap(data, 
                         cmap='viridis',
                         xticklabels=50,  # 每50个时间步显示一个标签
                         yticklabels=[f"{t:.2f}" for t in time_steps])
    
    plt.title('Output Values Over Time Steps and Sequence Length')
    plt.xlabel('Sequence Position')
    plt.ylabel('Time Step')
    
    # 使用返回的heatmap对象添加colorbar
    plt.colorbar(heatmap.collections[0], label='Mean Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved as {save_path}")
    
    # 创建统计图
    plt.figure(figsize=(15, 5))

def main():
    # 加载新版本的预测
    predictions = load_predictions("./out/data/new")
    
    # 创建热力图
    create_heatmap(predictions, 'new_model_heatmap.png')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()