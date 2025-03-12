import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.nn.functional import cosine_similarity
from kneed import KneeLocator

# 设置目录路径
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ff_output')

def load_and_organize_data():
    """加载所有pt文件并按块号和时间步组织"""
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误：目录 {data_dir} 不存在")
        return None
    
    # 获取所有pt文件
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    if not pt_files:
        print(f"错误：在 {data_dir} 中未找到pt文件")
        return None
    
    # 按块号组织数据
    data_by_block = defaultdict(dict)
    
    for file_path in pt_files:
        # 解析文件名
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        
        # 假设文件名格式为 "块号_时间步号.pt"
        parts = file_name_without_ext.split('_')
        if len(parts) < 2:
            print(f"警告：文件名 {file_name} 格式不正确，应为'块号_时间步号.pt'")
            continue
            
        block_idx = int(parts[0])
        step_idx = int(parts[1])
        
        # 加载张量数据
        try:
            tensor_data = torch.load(file_path)
            data_by_block[block_idx][step_idx] = tensor_data
        except Exception as e:
            print(f"错误：无法加载文件 {file_name}：{e}")
    
    return data_by_block

def calculate_adjacent_steps_similarity(data_by_block):
    """计算每个块中相邻时间步的输出相似度"""
    if data_by_block is None:
        return None
    
    similarities = []
    
    for block_idx, steps_data in data_by_block.items():
        # 获取排序后的时间步列表
        step_indices = sorted(steps_data.keys())
        
        # 计算相邻时间步之间的相似度
        for i in range(len(step_indices) - 1):
            current_step = step_indices[i]
            next_step = step_indices[i + 1]
            
            current_tensor = steps_data[current_step]
            next_tensor = steps_data[next_step]
            
            # 确保形状兼容
            if current_tensor.shape != next_tensor.shape:
                print(f"警告：块 {block_idx} 中时间步 {current_step} 和 {next_step} 的张量形状不同")
                continue
            
            # 将张量展平用于计算余弦相似度
            current_flat = current_tensor.reshape(current_tensor.shape[0], -1)
            next_flat = next_tensor.reshape(next_tensor.shape[0], -1)
            
            # 按批次计算余弦相似度并取平均值
            sim = cosine_similarity(current_flat, next_flat, dim=1).mean().item()
            similarities.append((block_idx, current_step, next_step, sim))
    
    return similarities

def calculate_cond_uncond_similarity(data_by_block):
    """计算每个块中条件和非条件部分的相似度"""
    if data_by_block is None:
        return None
    
    similarities = []
    
    for block_idx, steps_data in data_by_block.items():
        for step_idx, tensor_data in steps_data.items():
            # 确保张量可以在第一维(dim=0)切分为两块
            if tensor_data.shape[0] % 2 != 0:
                print(f"警告：块 {block_idx} 时间步 {step_idx} 的张量不能在dim=0上平均切分")
                continue
            
            # 切分为条件和非条件部分
            x_cond, x_uncond = tensor_data.chunk(2, dim=0)
            
            # 将张量展平用于计算余弦相似度
            x_cond_flat = x_cond.reshape(x_cond.shape[0], -1)
            x_uncond_flat = x_uncond.reshape(x_uncond.shape[0], -1)
            
            # 计算余弦相似度并取平均值
            sim = cosine_similarity(x_cond_flat, x_uncond_flat, dim=1).mean().item()
            similarities.append((block_idx, step_idx, sim))
    
    return similarities

def find_threshold_with_kneedle(data, curve='convex', direction='decreasing'):
    """使用kneedle方法找到阈值"""
    if not data or len(data) < 3:
        return np.mean(data) if data else 0
    
    # 排序数据
    sorted_data = np.sort(data)
    
    # 构造x和y数据
    y = sorted_data
    x = np.arange(len(sorted_data))
    
    try:
        # 使用KneeLocator找到阈值
        kneedle = KneeLocator(x, y, curve=curve, direction=direction)
        threshold = kneedle.knee_y
        
        # 如果没有找到阈值，使用平均值
        if threshold is None:
            threshold = np.mean(data)
    except Exception as e:
        print(f"警告：使用kneedle方法找到阈值失败：{e}")
        threshold = np.mean(data)
    
    return threshold

def plot_histograms(adjacent_similarities, cond_uncond_similarities):
    """绘制相似度直方图并找到阈值"""
    plt.figure(figsize=(15, 6))
    
    # 相邻时间步相似度直方图
    plt.subplot(1, 2, 1)
    adjacent_threshold = None
    if adjacent_similarities:
        similarities = [sim for _, _, _, sim in adjacent_similarities]
        plt.hist(similarities, bins=20, alpha=0.7, color='blue')
        
        # 使用kneedle方法找到阈值
        adjacent_threshold = find_threshold_with_kneedle(similarities)
        
        # 绘制阈值线
        plt.axvline(adjacent_threshold, color='red', linestyle='dashed', linewidth=2)
        plt.title(f'相邻时间步输出相似度分布 (阈值: {adjacent_threshold:.4f})')
    else:
        plt.title('无相邻时间步相似度数据')
    plt.xlabel('余弦相似度')
    plt.ylabel('频次')
    plt.grid(True, alpha=0.3)
    
    # 条件与非条件相似度直方图
    plt.subplot(1, 2, 2)
    cond_uncond_threshold = None
    if cond_uncond_similarities:
        similarities = [sim for _, _, sim in cond_uncond_similarities]
        plt.hist(similarities, bins=20, alpha=0.7, color='green')
        
        # 使用kneedle方法找到阈值
        cond_uncond_threshold = find_threshold_with_kneedle(similarities)
        
        # 绘制阈值线
        plt.axvline(cond_uncond_threshold, color='red', linestyle='dashed', linewidth=2)
        plt.title(f'条件与非条件分支相似度分布 (阈值: {cond_uncond_threshold:.4f})')
    else:
        plt.title('无条件与非条件相似度数据')
    plt.xlabel('余弦相似度')
    plt.ylabel('频次')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = os.path.dirname(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'ff_similarity_analysis.png'), dpi=300)
    print(f"图像已保存到 {os.path.join(output_dir, 'ff_similarity_analysis.png')}")
    
    # 显示图像
    plt.show()
    
    return adjacent_threshold, cond_uncond_threshold

def plot_above_threshold_matrix(cond_uncond_similarities, cond_uncond_threshold):
    """绘制超过阈值的条件与非条件相似度矩阵"""
    if not cond_uncond_similarities or cond_uncond_threshold is None:
        print("无数据或阈值未定义，无法绘制矩阵")
        return
    
    # 获取超过阈值的数据
    above_threshold = [(block_idx, step_idx, sim) for block_idx, step_idx, sim in cond_uncond_similarities if sim > cond_uncond_threshold]
    
    if not above_threshold:
        print("无超过阈值的数据，无法绘制矩阵")
        return
    
    # 获取所有块号和时间步号
    block_indices = sorted(set([block_idx for block_idx, _, _ in cond_uncond_similarities]))
    step_indices = sorted(set([step_idx for _, step_idx, _ in cond_uncond_similarities]))
    
    # 创建矩阵
    matrix = np.zeros((len(block_indices), len(step_indices)))
    
    # 创建块号和时间步号的映射
    block_idx_map = {idx: i for i, idx in enumerate(block_indices)}
    step_idx_map = {idx: i for i, idx in enumerate(step_indices)}
    
    # 填充矩阵
    for block_idx, step_idx, sim in cond_uncond_similarities:
        matrix[block_idx_map[block_idx], step_idx_map[step_idx]] = sim
    
    # 绘制矩阵
    plt.figure(figsize=(12, 8))
    im = plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='余弦相似度')
    plt.xlabel('时间步')
    plt.ylabel('块号')
    plt.title('条件与非条件相似度矩阵')
    plt.xticks(range(len(step_indices)), step_indices)
    plt.yticks(range(len(block_indices)), block_indices)
    plt.grid(False)
    
    # 绘制超过阈值的点
    for block_idx, step_idx, sim in above_threshold:
        i = block_idx_map[block_idx]
        j = step_idx_map[step_idx]
        plt.scatter(j, i, color='red', s=100, marker='o', edgecolors='white', linewidths=1.5)
    
    # 添加图例
    plt.scatter([], [], color='red', s=100, marker='o', edgecolors='white', linewidths=1.5, label=f'超过阈值 ({cond_uncond_threshold:.4f})')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = os.path.dirname(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'ff_similarity_matrix.png'), dpi=300)
    print(f"图像已保存到 {os.path.join(output_dir, 'ff_similarity_matrix.png')}")
    
    # 显示图像
    plt.show()

def analyze_by_block(adjacent_similarities, cond_uncond_similarities, adjacent_threshold=None, cond_uncond_threshold=None):
    """按块分析相似度并输出超过阈值的数据"""
    if not adjacent_similarities and not cond_uncond_similarities:
        return
    
    # 如果没有提供阈值，使用平均值
    if adjacent_threshold is None and adjacent_similarities:
        similarities = [sim for _, _, _, sim in adjacent_similarities]
        adjacent_threshold = np.mean(similarities)
    
    if cond_uncond_threshold is None and cond_uncond_similarities:
        similarities = [sim for _, _, sim in cond_uncond_similarities]
        cond_uncond_threshold = np.mean(similarities)
    
    # 按块组织相邻时间步相似度
    adjacent_by_block = defaultdict(list)
    for block_idx, step1, step2, sim in adjacent_similarities:
        adjacent_by_block[block_idx].append((step1, step2, sim))
    
    # 按块组织条件与非条件相似度
    cond_uncond_by_block = defaultdict(list)
    for block_idx, step_idx, sim in cond_uncond_similarities:
        cond_uncond_by_block[block_idx].append((step_idx, sim))
    
    # 输出每个块的平均相似度
    print("\n按块分析相似度:")
    print("-" * 60)
    print(f"{'块号':^10}|{'相邻时间步平均相似度':^25}|{'条件-非条件平均相似度':^25}")
    print("-" * 60)
    
    all_blocks = sorted(set(list(adjacent_by_block.keys()) + list(cond_uncond_by_block.keys())))
    for block_idx in all_blocks:
        adjacent_avg = np.mean([sim for _, _, sim in adjacent_by_block[block_idx]]) if block_idx in adjacent_by_block else float('nan')
        cond_uncond_avg = np.mean([sim for _, sim in cond_uncond_by_block[block_idx]]) if block_idx in cond_uncond_by_block else float('nan')
        print(f"{block_idx:^10}|{adjacent_avg:^25.4f}|{cond_uncond_avg:^25.4f}")
    
    # 输出超过阈值的相邻时间步数据
    if adjacent_threshold is not None and adjacent_similarities:
        print("\n超过阈值的相邻时间步数据:")
        print("-" * 60)
        print(f"{'块号':^10}|{'时间步1':^10}|{'时间步2':^10}|{'相似度':^15}")
        print("-" * 60)
        
        above_threshold = [(block_idx, step1, step2, sim) for block_idx, step1, step2, sim in adjacent_similarities if sim > adjacent_threshold]
        above_threshold.sort(key=lambda x: x[3], reverse=True)  # 按相似度降序排序
        
        for block_idx, step1, step2, sim in above_threshold:
            print(f"{block_idx:^10}|{step1:^10}|{step2:^10}|{sim:^15.4f}")
    
    # 输出超过阈值的条件与非条件数据
    if cond_uncond_threshold is not None and cond_uncond_similarities:
        print("\n超过阈值的条件与非条件数据:")
        print("-" * 50)
        print(f"{'块号':^10}|{'时间步':^10}|{'相似度':^15}")
        print("-" * 50)
        
        above_threshold = [(block_idx, step_idx, sim) for block_idx, step_idx, sim in cond_uncond_similarities if sim > cond_uncond_threshold]
        above_threshold.sort(key=lambda x: x[2], reverse=True)  # 按相似度降序排序
        
        for block_idx, step_idx, sim in above_threshold:
            print(f"{block_idx:^10}|{step_idx:^10}|{sim:^15.4f}")

def main():
    print("开始分析前馈网络输出的相似度...")
    print(f"数据目录: {data_dir}")
    
    # 加载并组织数据
    data_by_block = load_and_organize_data()
    if data_by_block is None:
        return
    
    print(f"已加载 {len(data_by_block)} 个块的数据")
    
    # 计算相似度
    adjacent_similarities = calculate_adjacent_steps_similarity(data_by_block)
    cond_uncond_similarities = calculate_cond_uncond_similarity(data_by_block)
    
    # 绘制直方图并找到阈值
    adjacent_threshold, cond_uncond_threshold = plot_histograms(adjacent_similarities, cond_uncond_similarities)
    
    # 绘制超过阈值的条件与非条件相似度矩阵
    plot_above_threshold_matrix(cond_uncond_similarities, cond_uncond_threshold)
    
    # # 按块分析相似度并输出超过阈值的数据
    # analyze_by_block(adjacent_similarities, cond_uncond_similarities, adjacent_threshold, cond_uncond_threshold)
    
    print("分析完成!")

if __name__ == "__main__":
    main()
