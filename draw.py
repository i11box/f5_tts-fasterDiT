import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 定义基准FLOPS（无加速）
FLOPS_DATA_10 = {
    0.05: {'attention_gflops': 59.661068543999995},
    0.10: {'attention_gflops': 58.181776576},
    0.15: {'attention_gflops': 55.353881968},
    0.20: {'attention_gflops': 51.27748718400001},
    0.25: {'attention_gflops': 46.934152495999996},
    0.30: {'attention_gflops': 44.909858224000004}
}

BASE_FLOPS_10 = {'attention_gflops': 60.684338175999976}  # 无加速的基准数据

FLOPS_DATA_70 = {
    0.05: {'attention_gflops': 177.69089923199996},
    0.10: {'attention_gflops': 173.244693888},
    0.15: {'attention_gflops': 164.824226784},
    0.20: {'attention_gflops': 152.68616899200003},
    0.25: {'attention_gflops': 139.753258848},
    0.30: {'attention_gflops': 133.725628512}
}

BASE_FLOPS_70 = {'attention_gflops': 180.69643468799993}  # 无加速的基准数据

def plot_combined_analysis(json_file, delta, text_length=10):
    fig = plt.figure(figsize=(20, 8))
    
    # 左侧热力图
    ax1 = plt.subplot(121)
    plot_strategy_heatmap(json_file, delta, ax1)
    
    # 右侧加速比曲线
    ax2 = plt.subplot(122)
    plot_speedup_curves(text_length, ax2)
    
    plt.tight_layout()
    plt.savefig(f'combined_analysis_{text_length}chars_delta{delta:.2f}.png')
    plt.close()

def plot_strategy_heatmap(json_file, delta, ax):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    strategy_dict = data
    block_ids = sorted([int(bid) for bid in strategy_dict.keys()])
    timesteps = sorted([float(t) for t in strategy_dict['0'].keys()])
    
    strategy_matrix = np.zeros((len(timesteps), len(block_ids)), dtype=int)
    strategies = ['none', 'ast', 'asc', 'wars', 'asc-wars']
    strategy_to_num = {strategy: i for i, strategy in enumerate(strategies)}
    
    for j, bid in enumerate(block_ids):
        block_strategies = strategy_dict[str(bid)]
        for i, t in enumerate(timesteps):
            strategy = block_strategies[f'{t:.3f}']
            if strategy in strategy_to_num:
                strategy_matrix[i, j] = strategy_to_num[strategy]
    
    colors = ['#F5F9FF', '#99C7FF', '#3388FF', '#0066FF', '#0000FF']
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm(boundaries=range(len(strategies) + 1), ncolors=len(strategies))
    
    sns.heatmap(strategy_matrix, 
                cmap=cmap,
                norm=norm,
                xticklabels=block_ids,
                yticklabels=[f'{t:.3f}' for t in timesteps],
                cbar_kws={'ticks': np.arange(len(strategies)) + 0.5,
                         'boundaries': range(len(strategies) + 1)},
                ax=ax)
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticklabels(strategies)
    
    ax.set_xlabel('Block ID')
    ax.set_ylabel('Timestep')
    ax.set_title(f'Strategy Distribution (delta={delta:.2f})')

def plot_speedup_curves(text_length, ax):
    data = FLOPS_DATA_10 if text_length == 10 else FLOPS_DATA_70
    base = BASE_FLOPS_10 if text_length == 10 else BASE_FLOPS_70
    
    deltas = sorted(data.keys())
    speedups_total = [base['attention_gflops'] / data[d]['attention_gflops'] for d in deltas]
    
    # 使用更粗的线条和更大的标记
    ax.plot(deltas, speedups_total, 'o-', label='Speedup Ratio', 
            linewidth=3, markersize=8, color='#3388FF')
    
    # 添加数值标签
    for x, y in zip(deltas, speedups_total):
        ax.text(x, y, f'{y:.2f}x', 
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Delta (Compression Threshold)')
    ax.set_ylabel('Speedup Ratio')
    ax.set_title(f'Speedup Analysis ({text_length} chars)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置y轴范围，留出标签空间
    ax.set_ylim(min(speedups_total) * 0.9, max(speedups_total) * 1.1)
    
    # 可选：美化x轴刻度
    ax.set_xticks(deltas)
    ax.set_xticklabels([f'{d:.2f}' for d in deltas])

if __name__ == '__main__':
    deltas = [0.05,0.1, 0.15, 0.2, 0.25, 0.3]
    for delta in deltas:
        for text_length in [10, 70]:
            json_file = 'method' + str(delta) + '.json'
            if os.path.exists(json_file):
                plot_combined_analysis(json_file, delta, text_length)
            else:
                print(f"File not found: {json_file}")