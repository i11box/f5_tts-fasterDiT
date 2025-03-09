import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_similarity_heatmap(file_path, threshold=0.01, output_path=None):
    """
    读取相似度文件并绘制块ID和步骤ID的热力图
    
    Args:
        file_path: 相似度文件路径
        threshold: 相似度阈值，只有大于此值的组合才会被标记为选中
        output_path: 输出图像的路径
    """
    # 读取文件
    data = []
    with open(file_path, 'r') as f:
        # 跳过标题行
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                block_id = int(parts[0])
                step_id = int(parts[1])
                similarity = float(parts[2])
                data.append((block_id, step_id, similarity))
    
    if not data:
        print(f"警告: 文件 {file_path} 中没有找到有效数据")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(data, columns=['block_id', 'step_id', 'similarity'])
    
    # 找出最大的块ID和步骤ID，确定热力图的尺寸
    max_block_id = df['block_id'].max()
    max_step_id = df['step_id'].max()
    
    # 创建一个空的矩阵，用于存储相似度值
    # 注意：我们加1是因为ID从0开始
    heatmap_data = np.zeros((max_block_id + 1, max_step_id + 1))
    
    # 填充矩阵
    for _, row in df.iterrows():
        block_id = int(row['block_id'])
        step_id = int(row['step_id'])
        similarity = float(row['similarity'])
        heatmap_data[block_id, step_id] = similarity
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 使用seaborn绘制热力图
    ax = sns.heatmap(heatmap_data, cmap='viridis', 
                    vmin=0, vmax=1,
                    cbar_kws={'label': '余弦相似度'})
    
    # 设置标题和标签
    plt.title('块ID和步骤ID的相似度热力图', fontsize=16)
    plt.xlabel('步骤ID', fontsize=12)
    plt.ylabel('块ID', fontsize=12)
    
    # 调整坐标轴刻度
    # 只显示整数刻度
    ax.set_xticks(np.arange(0, max_step_id + 1, 5))
    ax.set_yticks(np.arange(0, max_block_id + 1, 2))
    ax.set_xticklabels(np.arange(0, max_step_id + 1, 5))
    ax.set_yticklabels(np.arange(0, max_block_id + 1, 2))
    
    # 添加阈值线
    if threshold > 0:
        # 找出高于阈值的块和步骤组合
        high_sim_blocks = df[df['similarity'] >= threshold]
        
        # 在热力图上标记这些组合
        for _, row in high_sim_blocks.iterrows():
            block_id = int(row['block_id'])
            step_id = int(row['step_id'])
            plt.plot(step_id + 0.5, block_id + 0.5, 'ro', markersize=3)
    
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到: {output_path}")
    
    plt.show()

def main():
    # 文件路径
    file_paths = [
        "high_sim_otsu_combinations.txt",
        "high_sim_knee_combinations.txt"
    ]
    
    # 检查文件是否存在
    existing_files = [f for f in file_paths if os.path.exists(f)]
    
    if not existing_files:
        # 尝试查找当前目录下的文件
        current_dir_files = [f for f in os.listdir("../") if f.startswith("high_sim") and f.endswith(".txt")]
        if current_dir_files:
            existing_files = [os.path.join("../", f) for f in current_dir_files]
        else:
            print("错误: 找不到相似度文件")
            return
    
    # 为每个文件生成热力图
    for file_path in existing_files:
        print(f"处理文件: {file_path}")
        output_name = os.path.basename(file_path).replace(".txt", "_heatmap.png")
        output_path = os.path.join("../", output_name)
        
        # 默认阈值设为0.9，可以根据需要调整
        plot_similarity_heatmap(file_path, threshold=0.01, output_path=output_path)

# 添加一个函数用于比较两种方法的结果
def compare_methods(otsu_file, knee_file, output_path=None):
    """
    比较Otsu方法和拐点方法的结果
    
    Args:
        otsu_file: Otsu方法结果文件路径
        knee_file: 拐点方法结果文件路径
        output_path: 输出图像的路径
    """
    # 读取两个文件
    otsu_data = []
    knee_data = []
    
    # 读取Otsu文件
    with open(otsu_file, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                block_id = int(parts[0])
                step_id = int(parts[1])
                otsu_data.append((block_id, step_id))
    
    # 读取拐点文件
    with open(knee_file, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                block_id = int(parts[0])
                step_id = int(parts[1])
                knee_data.append((block_id, step_id))
    
    # 转换为集合，方便比较
    otsu_set = set(otsu_data)
    knee_set = set(knee_data)
    
    # 计算交集和差集
    common = otsu_set.intersection(knee_set)
    only_otsu = otsu_set - knee_set
    only_knee = knee_set - otsu_set
    
    print(f"Otsu方法选中的组合数: {len(otsu_set)}")
    print(f"拐点方法选中的组合数: {len(knee_set)}")
    print(f"两种方法共同选中的组合数: {len(common)}")
    print(f"只被Otsu方法选中的组合数: {len(only_otsu)}")
    print(f"只被拐点方法选中的组合数: {len(only_knee)}")
    
    # 找出最大的块ID和步骤ID
    all_block_ids = [block for block, _ in otsu_set.union(knee_set)]
    all_step_ids = [step for _, step in otsu_set.union(knee_set)]
    
    if not all_block_ids or not all_step_ids:
        print("警告: 没有找到有效数据进行比较")
        return
    
    max_block_id = max(all_block_ids)
    max_step_id = max(all_step_ids)
    
    # 创建比较矩阵
    # 0: 未选中, 1: 只被Otsu选中, 2: 只被拐点选中, 3: 两种方法都选中
    comparison_matrix = np.zeros((max_block_id + 1, max_step_id + 1))
    
    for block, step in only_otsu:
        comparison_matrix[block, step] = 1
    
    for block, step in only_knee:
        comparison_matrix[block, step] = 2
    
    for block, step in common:
        comparison_matrix[block, step] = 3
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 自定义颜色映射
    cmap = plt.cm.colors.ListedColormap(['white', 'red', 'blue', 'purple'])
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # 绘制热力图
    ax = sns.heatmap(comparison_matrix, cmap=cmap, norm=norm, 
                    cbar_kws={'label': '选中方法', 'ticks': [0.25, 1, 2, 3]})
    
    # 设置colorbar标签
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticklabels(['未选中', '只被Otsu选中', '只被拐点选中', '两种方法都选中'])
    
    # 设置标题和标签
    plt.title('Otsu方法与拐点方法比较', fontsize=16)
    plt.xlabel('步骤ID', fontsize=12)
    plt.ylabel('块ID', fontsize=12)
    
    # 调整坐标轴刻度
    ax.set_xticks(np.arange(0, max_step_id + 1, 5))
    ax.set_yticks(np.arange(0, max_block_id + 1, 2))
    ax.set_xticklabels(np.arange(0, max_step_id + 1, 5))
    ax.set_yticklabels(np.arange(0, max_block_id + 1, 2))
    
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"比较热力图已保存到: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
    
    # 如果两种方法的文件都存在，也进行比较
    otsu_file = "../high_sim_otsu_combinations.txt"
    knee_file = "../high_sim_knee_combinations.txt"
    
    if os.path.exists(otsu_file) and os.path.exists(knee_file):
        print("\n比较Otsu方法和拐点方法的结果:")
        compare_methods(otsu_file, knee_file, "../method_comparison_heatmap.png")
