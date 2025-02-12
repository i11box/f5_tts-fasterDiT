import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import re

def load_attention_outputs(data_dir: str):
    """加载所有注意力输出文件"""
    attention_files = glob(os.path.join(data_dir, "*.pt"))
    outputs = {}
    
    for file in attention_files:
        key = os.path.basename(file).replace(".pt", "")
        outputs[key] = torch.load(file)
    
    return outputs

def parse_key(key: str):
    """解析文件名以获取时间步、块ID和条件信息"""
    match = re.match(r"t(\d+\.\d+)_b(\d+)_(\w+)", key)
    if match:
        return float(match.group(1)), int(match.group(2)), match.group(3)
    raise ValueError(f"Invalid key format: {key}")

def compute_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """计算两个张量之间的余弦相似度"""
    flat1 = tensor1.reshape(-1)
    flat2 = tensor2.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
    return cos_sim.item()

def plot_similarity_heatmaps(data_dir: str = "./attention_outputs"):
    """绘制三张相似度热力图"""
    # 加载数据
    outputs = load_attention_outputs(data_dir)
    
    # 获取所有时间步和块ID
    timesteps = sorted(list(set(parse_key(k)[0] for k in outputs.keys())))
    block_ids = sorted(list(set(parse_key(k)[1] for k in outputs.keys())))
    
    # 选择第一个时间步和块号用于前两张图
    target_timestep = timesteps[20]
    target_block = block_ids[20]
    
    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. 块间相似度热力图
    blocks_at_timestep = [(k, v) for k, v in outputs.items() 
                         if parse_key(k)[0] == target_timestep and parse_key(k)[2] == "cond"]
    blocks_at_timestep.sort(key=lambda x: parse_key(x[0])[1])
    
    n_blocks = len(blocks_at_timestep)
    block_sim_matrix = np.zeros((n_blocks, n_blocks))
    block_labels = []
    
    for i, (key1, tensor1) in enumerate(blocks_at_timestep):
        block_labels.append(parse_key(key1)[1])
        for j, (key2, tensor2) in enumerate(blocks_at_timestep):
            block_sim_matrix[i, j] = compute_cosine_similarity(tensor1, tensor2)
    
    sns.heatmap(block_sim_matrix, ax=ax1, annot=False, fmt=".2f",
                xticklabels=block_labels, yticklabels=block_labels)
    ax1.set_title(f"Block Similarity at t={target_timestep:.3f}")
    ax1.set_xlabel("Block ID")
    ax1.set_ylabel("Block ID")
    
    # 2. 时间步相似度热力图
    timesteps_at_block = [(k, v) for k, v in outputs.items() 
                         if parse_key(k)[1] == target_block and parse_key(k)[2] == "cond"]
    timesteps_at_block.sort(key=lambda x: parse_key(x[0])[0])
    
    n_timesteps = len(timesteps_at_block)
    time_sim_matrix = np.zeros((n_timesteps, n_timesteps))
    time_labels = []
    
    for i, (key1, tensor1) in enumerate(timesteps_at_block):
        time_labels.append(f"{parse_key(key1)[0]:.3f}")
        for j, (key2, tensor2) in enumerate(timesteps_at_block):
            time_sim_matrix[i, j] = compute_cosine_similarity(tensor1, tensor2)
    
    sns.heatmap(time_sim_matrix, ax=ax2, annot=False, fmt=".2f",
                xticklabels=time_labels, yticklabels=time_labels)
    ax2.set_title(f"Timestep Similarity at Block {target_block}")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Timestep")
    
    # 3. 条件/非条件相似度热力图
    cond_sim_matrix = np.zeros((len(block_ids), len(timesteps)))
    
    for i, block_id in enumerate(block_ids):
        for j, timestep in enumerate(timesteps):
            cond_key = f"t{timestep:.3f}_b{block_id}_cond"
            null_key = f"t{timestep:.3f}_b{block_id}_null"
            if cond_key in outputs and null_key in outputs:
                cond_sim_matrix[i, j] = compute_cosine_similarity(
                    outputs[cond_key], outputs[null_key]
                )
    
    sns.heatmap(cond_sim_matrix, ax=ax3, annot=False, fmt=".2f",
                xticklabels=[f"{t:.3f}" for t in timesteps],
                yticklabels=block_ids)
    ax3.set_title("Condition vs Non-condition Similarity")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Block ID")
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("attention_similarity_analysis_3.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_similarity_heatmaps()
