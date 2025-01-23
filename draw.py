import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 定义基准FLOPS（无加速）
BASE_FLOPS_10 = {
    'total_gflops': 6383.761002431997,
    'attention_gflops': 2544.9494322559995,
    'linear_gflops': 3838.8115701759984
}

BASE_FLOPS_70 = {
    'total_gflops': 33995.086532352,
    'attention_gflops': 22564.467281663994,
    'linear_gflops': 11430.619250688005
}

# 定义加速后的FLOPS数据
FLOPS_DATA_10 = {
    0.10: {'total_gflops': 6336.959235551998, 'attention_gflops': 2526.291445216, 'linear_gflops': 3810.6677903359987},
    0.15: {'total_gflops': 5588.130965471999, 'attention_gflops': 2227.7636525760004, 'linear_gflops': 3360.3673128959995},
    0.20: {'total_gflops': 4586.573154239999, 'attention_gflops': 1828.4827299199997, 'linear_gflops': 2758.0904243200002},
    0.25: {'total_gflops': 3921.9880645439994, 'attention_gflops': 1563.5393139520002, 'linear_gflops': 2358.4487505919997},
    0.30: {'total_gflops': 3341.646155232, 'attention_gflops': 1332.1802746560002, 'linear_gflops': 2009.465880576},
    0.35: {'total_gflops': 2954.1519648959993, 'attention_gflops': 1169.83632304, 'linear_gflops': 1784.315641856},
    0.40: {'total_gflops': 2617.17924336, 'attention_gflops': 1035.498816352, 'linear_gflops': 1581.680427008}
}

FLOPS_DATA_70 = {
    0.10: {'total_gflops': 33745.855692672, 'attention_gflops': 22399.038635903995, 'linear_gflops': 11346.817056768004},
    0.15: {'total_gflops': 29758.162257792, 'attention_gflops': 19752.180303743997, 'linear_gflops': 10005.981954047998},
    0.20: {'total_gflops': 24424.622288639996, 'attention_gflops': 16212.007284479998, 'linear_gflops': 8212.61500416},
    0.25: {'total_gflops': 20885.544365183996, 'attention_gflops': 13862.920514687996, 'linear_gflops': 7022.623850496001},
    0.30: {'total_gflops': 17795.081953152, 'attention_gflops': 11811.605307263997, 'linear_gflops': 5983.476645887999},
    0.35: {'total_gflops': 15685.385505408003, 'attention_gflops': 10372.32641088, 'linear_gflops': 5313.059094528001},
    0.40: {'total_gflops': 13890.923459712, 'attention_gflops': 9181.240161407999, 'linear_gflops': 4709.683298304}
}

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
    
    strategy_dict = data.get('strategies', {})
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

def plot_speedup_curves(text_length, save_path=None):
    data = FLOPS_DATA_10 if text_length == 10 else FLOPS_DATA_70
    base = BASE_FLOPS_10 if text_length == 10 else BASE_FLOPS_70
    
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    deltas = sorted(data.keys())
    speedups_total = [base['total_gflops'] / data[d]['total_gflops'] for d in deltas]
    
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
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    for text_length in [10, 70]:
        save_path = f'speedup_analysis_{text_length}chars.png'
        plot_speedup_curves(text_length, save_path)