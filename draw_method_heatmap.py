import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_strategy_heatmap(json_file):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取条件模型的策略
    strategy_dict = data['strategies']['cond']
    
    # 获取所有块ID和时间步
    block_ids = sorted([int(bid) for bid in strategy_dict.keys()])
    # 使用第一个块的时间步作为参考
    timesteps = sorted([float(t) for t in strategy_dict['0'].keys()])
    
    # 创建策略矩阵
    strategy_matrix = np.zeros((len(timesteps), len(block_ids)), dtype=int)
    
    # 固定的策略到数字的映射
    strategies = ['none', 'ast', 'asc', 'wars']
    strategy_to_num = {strategy: i for i, strategy in enumerate(strategies)}
    
    # 填充策略矩阵
    for j, bid in enumerate(block_ids):
        block_strategies = strategy_dict[str(bid)]
        for i, t in enumerate(timesteps):
            strategy = block_strategies[f'{t:.3f}']
            if strategy in strategy_to_num:  # 只处理四种基本策略
                strategy_matrix[i, j] = strategy_to_num[strategy]
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 固定的蓝色系渐变色映射
    colors = [
        '#F5F9FF',  # 最浅的蓝色 (none)
        '#99C7FF',  # 浅蓝色 (ast)
        '#3388FF',  # 中蓝色 (asc)
        '#0066FF'   # 深蓝色 (wars)
    ]
    cmap = plt.cm.colors.ListedColormap(colors)
    
    # 设置标准化范围，确保颜色映射固定
    norm = plt.cm.colors.BoundaryNorm(boundaries=range(len(strategies) + 1), 
                                     ncolors=len(strategies))
    
    # 绘制热力图
    sns.heatmap(strategy_matrix, 
                cmap=cmap,
                norm=norm,
                xticklabels=block_ids,
                yticklabels=[f'{t:.3f}' for t in timesteps],
                cbar_kws={
                    'ticks': np.arange(len(strategies)) + 0.5,
                    'boundaries': range(len(strategies) + 1)
                })
    
    # 设置colorbar标签
    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticklabels(strategies)
    
    # 设置标签和标题
    plt.xlabel('Block ID')
    plt.ylabel('Timestep')
    plt.title('Strategy Distribution Heatmap (Conditional Model)')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.show()
    plt.close()

# 使用示例
if __name__ == '__main__':
    # 绘制条件模型的策略热力图
    plot_strategy_heatmap('model/backbones/method_cond_0.2.json')