import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def visualize_methods_advanced(json_file):
    """
    将JSON文件中的时间步和块号关系可视化为美观的热图
    
    Args:
        json_file: JSON文件路径
    """
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取所有时间步和块号
    steps = sorted([int(step) for step in data.keys()])
    all_blocks = set()
    for blocks in data.values():
        all_blocks.update([int(block) for block in blocks])
    blocks = sorted(list(all_blocks))
    
    # 创建一个矩阵来表示数据
    matrix = np.zeros((len(steps), len(blocks)))
    
    # 填充矩阵
    for i, step in enumerate(steps):
        for j, block in enumerate(blocks):
            if str(block) in data[str(step)]:
                matrix[i, j] = 1
    
    # 设置样式
    sns.set(style="whitegrid")
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 创建热图
    ax = sns.heatmap(matrix, 
                     cmap="YlGnBu", 
                     linewidths=0.5, 
                     linecolor='gray',
                     cbar_kws={'label': '存在关系', 'ticks': [0, 1]})
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(blocks)) + 0.5)
    ax.set_yticks(np.arange(len(steps)) + 0.5)
    ax.set_xticklabels(blocks)
    ax.set_yticklabels(steps)
    
    # 添加标题和标签
    plt.title('F5-TTS优化策略分布图', fontsize=18, pad=20)
    plt.xlabel('块号 (Block)', fontsize=14, labelpad=10)
    plt.ylabel('时间步 (Step)', fontsize=14, labelpad=10)
    
    # 添加注释
    total_optimized = np.sum(matrix)
    total_possible = matrix.size
    optimization_rate = total_optimized / total_possible * 100
    plt.figtext(0.5, 0.01, 
                f'优化覆盖率: {optimization_rate:.1f}% ({int(total_optimized)}/{total_possible}个组合)',
                ha='center', fontsize=12)
    
    # 保存图片
    output_file = os.path.splitext(json_file)[0] + '_advanced_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存到 {output_file}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # JSON文件路径
    json_file = os.path.join(current_dir, "selected_methods.json")
    
    # 可视化数据
    visualize_methods_advanced(json_file)