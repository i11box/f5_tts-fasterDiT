import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_loss_heatmap(json_path, save_dir):
    """
    将json文件中的loss数据可视化为热力图
    
    Args:
        json_path: json文件的路径
        save_dir: 保存图片的目录
    """
    # 读取json文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取block和step的数量
    blocks = sorted([int(k.split('_')[1]) for k in data.keys()])
    steps = sorted([int(k.split('_')[1]) for k in data[list(data.keys())[0]].keys()])
    
    # 创建loss矩阵
    loss_matrix = np.zeros((len(blocks), len(steps)))
    for i, block in enumerate(blocks):
        for j, step in enumerate(steps):
            loss_matrix[i, j] = data[f'block_{block}'][f'step_{step}']
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    sns.heatmap(loss_matrix, 
                cmap='viridis',
                xticklabels=steps,
                yticklabels=blocks,
                cbar_kws={'label': 'Loss值'})
    
    # 设置标题和轴标签
    method_name = os.path.splitext(os.path.basename(json_path))[0]
    plt.title(f'策略 {method_name} 的Loss分布')
    plt.xlabel('时间步')
    plt.ylabel('块号')
    
    # 保存图片
    save_path = os.path.join(save_dir, f'{method_name}_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'已保存热力图: {save_path}')

def main():
    # 创建保存目录
    save_dir = os.path.join('assets', 'loss_heatmaps')
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有json文件
    json_dir = os.path.join('data', 'method_evaluation')
    if not os.path.exists(json_dir):
        print(f'目录不存在: {json_dir}')
        return
        
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print(f'未找到json文件: {json_dir}')
        return
    
    # 处理每个json文件
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        try:
            plot_loss_heatmap(json_path, save_dir)
        except Exception as e:
            print(f'处理文件 {json_file} 时出错: {str(e)}')

if __name__ == '__main__':
    main()
