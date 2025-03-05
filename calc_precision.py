# -*- coding: utf-8 -*-

import os
import torch
import numpy as np

def precision_k(a, b, k):
    """计算两组索引的前k个元素的精确度"""
    precision_scores = []
    for batch_idx in range(a.shape[0]):
        # 直接接比较索引值，而不使用集合
        a_indices = a[batch_idx, :k].cpu().numpy().astype(int)
        b_indices = b[batch_idx, :k].cpu().numpy().astype(int)
        
        # 计算交集大小
        intersection_count = 0
        for idx in a_indices:
            if idx in b_indices:
                intersection_count += 1
        
        precision = intersection_count / k
        precision_scores.append(precision)
    return np.mean(precision_scores)

# 定义数据目录
data_dir = "data/cfg_explore"

# 确保输出文件存在的目录存在
os.makedirs(os.path.dirname('precision_scores.txt') or '.', exist_ok=True)

# 读取保存的索引
cfg_indices_path = os.path.join(data_dir, "cfg_indices.pt")
map_indices_path = os.path.join(data_dir, "map_indices.pt")

if not os.path.exists(cfg_indices_path) or not os.path.exists(map_indices_path):
    print(f"错误：找不到索引文件。请确保文件存在: {cfg_indices_path} 和 {map_indices_path}")
    exit(1)

# 加载索引数据
print(f"正在加载索引数据...")
cfg_indices_dict = torch.load(cfg_indices_path)
map_indices_dict = torch.load(map_indices_path)

# 写入标题
with open('precision_scores.txt', 'w') as f:
    f.write("Block ID | Timestep | Precision@500\n")
    f.write("---------|---------|--------------\n")

# 计算每个块和时间步的精确度
print(f"开始计算精确度...")
score_lst = []
for key in sorted(cfg_indices_dict.keys()):
    if key in map_indices_dict:
        # 解析键以获取块ID和时间步
        block_id, timestep = map(int, key.split('_'))
        
        # 获取对应的索引
        cfg_index = cfg_indices_dict[key]
        map_index = map_indices_dict[key]
        
        # 计算精确度
        try:
            precision_score = precision_k(cfg_index, map_index, 500)
            score_lst.append(precision_score)
            result = f'block {block_id}, timestep {timestep}, precision_k: {precision_score:.4f}\n'
            print(result.strip())
            
            # 将结果写入文件
            with open('precision_scores.txt', 'a') as f:
                f.write(result)
        except Exception as e:
            error_msg = f"计算精确度时出错 (block {block_id}, timestep {timestep}): {e}"
            print(error_msg)
            with open('precision_scores.txt', 'a') as f:
                f.write(f"block {block_id}, timestep {timestep}, ERROR: {e}\n")

print("\n结果已保存到 precision_scores.txt")
print(f'均值为{np.array(score_lst).mean()}')
