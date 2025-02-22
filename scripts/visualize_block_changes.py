import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def calculate_change_percentage(current, next_step):
    """计算相邻时间步的变化百分比"""
    # 确保输入张量在 GPU 上
    current = current.to('cuda')
    next_step = next_step.to('cuda')
    
    # 计算变化百分比
    change = torch.clamp(torch.abs((next_step - current) / (current + 1e-8)), max=0.1) * 100
    # 反转
    change = 10 - change
    return change.cpu().numpy()  # 转换回 CPU 以便后续处理

def plot_heatmap(change_matrix, block_id, step_id, output_dir, cond_or_uncond):
    """绘制热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(change_matrix, cmap='YlOrRd', xticklabels=False, yticklabels=False)
    plt.title(f'Block {block_id} - Step {step_id} to {step_id+1} Changes - {cond_or_uncond}')
    
    # 确保输出目录存在
    block_dir = os.path.join(output_dir, f'block_{block_id}')
    os.makedirs(block_dir, exist_ok=True)
    
    # 保存图片
    plt.savefig(os.path.join(block_dir, f'step_{step_id}_changes_{cond_or_uncond}.png'))
    plt.close()

def show_data(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap='YlOrRd', xticklabels=False, yticklabels=False)
    plt.show()
    

def visualize_block_changes(block_outputs_dir_src, block_outputs_dir_dst, output_dir):
    """主函数：处理数据并生成热力图"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有可用的 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 遍历 block_outputs 目录下的所有文件
    total_steps = 22 * 31
    with tqdm(total=total_steps, desc="Processing blocks") as pbar:
        for block_id in range(22):
            print(f"Processing block {block_id}")
            os.makedirs(os.path.join(output_dir, f'block_{block_id}'), exist_ok=True)
            for step in range(1,31):
                filename = f"block_{block_id}_step_{step}.pt"
                # 加载数据
                src_block_data = torch.load(os.path.join(block_outputs_dir_src, filename), map_location=device)
                dst_block_data = torch.load(os.path.join(block_outputs_dir_dst, filename), map_location=device)

                # 分为有条件和无条件的两部分
                src_block_data_cond , src_block_data_uncond = src_block_data[0], src_block_data[1]
                dst_block_data_cond , dst_block_data_uncond = dst_block_data[0], dst_block_data[1]

                # 转换为张量并确保在 GPU 上
                if isinstance(src_block_data, torch.Tensor):
                    src_block_data = src_block_data.to(device)
                if isinstance(dst_block_data, torch.Tensor):
                    dst_block_data = dst_block_data.to(device)
            
                # 计算变化百分比
                cond_change_matrix = calculate_change_percentage(src_block_data_cond, dst_block_data_cond)
                uncond_change_matrix = calculate_change_percentage(src_block_data_uncond, dst_block_data_uncond)
                
                # 条件和非条件分开保存
                cond_out_dir = os.path.join(output_dir, f'block_{block_id}','cond')
                uncond_out_dir = os.path.join(output_dir, f'block_{block_id}','uncond')
                os.makedirs(cond_out_dir, exist_ok=True)
                os.makedirs(uncond_out_dir, exist_ok=True)
                
                # 绘制并保存热力图
                plot_heatmap(cond_change_matrix, block_id, step, cond_out_dir, "cond")
                plot_heatmap(uncond_change_matrix, block_id, step, uncond_out_dir, "uncond")
                
                pbar.update(1)

if __name__ == "__main__":
    # 设置输入输出路径
    block_outputs_dir_src = "data/block_outputs_none"  # block_outputs 文件夹路径
    block_outputs_dir_dst = "data/block_outputs_0.2"
    output_dir = "assets/block_changes_compress_0.2"

    # 运行可视化
    visualize_block_changes(block_outputs_dir_src, block_outputs_dir_dst, output_dir)