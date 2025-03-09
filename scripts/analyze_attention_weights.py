import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.filters import threshold_otsu
import re
from kneed import KneeLocator

def calculate_cosine_similarity(attn_weights_dir):
    """计算注意力权重条件和非条件部分的余弦相似度
    
    Args:
        attn_weights_dir: 包含.pt文件的目录路径
        
    Returns:
        相似度的列表, 块和步骤信息的列表
    """
    # 获取所有.pt文件
    pt_files = glob(os.path.join(attn_weights_dir, "*.pt"))
    
    if not pt_files:
        raise ValueError(f"在{attn_weights_dir}目录下未找到.pt文件")
    
    # 存储每个文件的相似度
    similarities = []
    # 存储块和时间步信息
    block_step_info = []
    
    # 正则表达式提取块和步骤信息
    pattern = r"block_?(\d+).*step_?(\d+)"
    
    # 处理每个文件
    for pt_file in tqdm(pt_files, desc="计算余弦相似度"):
        # 从文件名中提取块和步骤信息
        file_name = os.path.basename(pt_file)
        match = re.search(pattern, file_name)
        
        if match:
            block_id = int(match.group(1))
            step_id = int(match.group(2))
        else:
            # 如果无法从文件名提取信息，报错
            raise ValueError("无法从文件名提取块和步骤信息")
        
        # 加载张量
        x = torch.load(pt_file)
        
        # 将张量分成两部分
        cond_part, uncond_part = x.chunk(2, dim=0)
        
        # 计算余弦相似度
        # 将矩阵展平为向量
        cond_flat = cond_part.reshape(-1)
        uncond_flat = uncond_part.reshape(-1)
        
        # 计算点积
        dot_product = torch.dot(cond_flat, uncond_flat)
        
        # 计算范数
        cond_norm = torch.norm(cond_flat)
        uncond_norm = torch.norm(uncond_flat)
        
        # 计算余弦相似度
        cosine_similarity = dot_product / (cond_norm * uncond_norm)
        
        # 添加到列表
        similarities.append(cosine_similarity.item())
        block_step_info.append((block_id, step_id, pt_file))
    
    return similarities, block_step_info

def apply_otsu_threshold(similarities, block_step_info):
    """应用Otsu阈值分割方法识别高于阈值的组合
    
    Args:
        similarities: 相似度列表
        block_step_info: 块和步骤信息列表
        
    Returns:
        阈值, 高于阈值的块和步骤组合
    """
    # 将相似度转换为numpy数组
    sim_array = np.array(similarities)
    
    # 计算Otsu阈值
    threshold = threshold_otsu(sim_array)
    # 计算kneedle阈值
    # threshold = calculate_threshold_kneedle(sim_array)
    # 计算CDF阈值
    # threshold = calculate_threshold_cdf_derivative(sim_array)
    
    # 找出高于阈值的索引
    above_threshold_indices = np.where(sim_array > threshold)[0]
    
    # 获取高于阈值的块和步骤组合
    above_threshold_combinations = [
        (block_step_info[i][0], block_step_info[i][1], similarities[i], block_step_info[i][2]) 
        for i in above_threshold_indices
    ]
    
    # 按相似度降序排序
    above_threshold_combinations.sort(key=lambda x: x[2], reverse=True)
    
    return threshold, above_threshold_combinations

def apply_knee_threshold(similarities, block_step_info):
    """应用拐点检测方法识别高于阈值的组合
    
    Args:
        similarities: 相似度列表
        block_step_info: 块和步骤信息列表
        
    Returns:
        拐点阈值, 高于阈值的块和步骤组合列表
    """
    # 将相似度转换为numpy数组并排序
    sim_array = np.array(similarities)
    sorted_sim = np.sort(sim_array)
    
    # 创建累积分布函数(CDF)
    y = np.arange(len(sorted_sim)) / float(len(sorted_sim))
    
    # 寻找拐点 (使用凹曲线和增加方向)
    try:
        knee = KneeLocator(sorted_sim, y, curve='concave', direction='increasing')
        threshold = knee.knee
        
        # 如果没有找到明显的拐点，使用数据的中位数作为备选
        if threshold is None:
            threshold = np.median(sorted_sim)
            print("警告: 未检测到明显拐点，使用中位数作为阈值")
    except Exception as e:
        print(f"拐点检测失败: {e}")
        threshold = np.median(sorted_sim)
        print("使用中位数作为备选阈值")
    
    # 找出高于阈值的索引
    above_threshold_indices = np.where(sim_array > threshold)[0]
    
    # 获取高于阈值的块和步骤组合
    above_threshold_combinations = [
        (block_step_info[i][0], block_step_info[i][1], similarities[i], block_step_info[i][2]) 
        for i in above_threshold_indices
    ]
    
    # 按相似度降序排序
    above_threshold_combinations.sort(key=lambda x: x[2], reverse=True)
    
    return threshold, above_threshold_combinations

def plot_similarity_histogram(similarities, threshold=None, knee_threshold=None, save_path=None):
    """绘制相似度的直方图
    
    Args:
        similarities: 相似度列表
        threshold: Otsu阈值（如果有）
        knee_threshold: 拐点阈值（如果有）
        save_path: 保存图像的路径(可选)
    """
    plt.figure(figsize=(10, 6))
    
    # 计算适当的bin数量
    n_bins = min(50, int(np.sqrt(len(similarities))))
    
    # 绘制直方图
    counts, bins, patches = plt.hist(similarities, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # 如果有阈值，绘制阈值线
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Otsu阈值: {threshold:.4f}')
        plt.legend()
    
    # 如果有拐点阈值，绘制拐点阈值线
    if knee_threshold is not None:
        plt.axvline(x=knee_threshold, color='g', linestyle='-.', label=f'拐点阈值: {knee_threshold:.4f}')
        plt.legend()
    
    # 添加标题和标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('注意力权重条件与非条件部分相似度分布', fontsize=14)
    plt.xlabel('余弦相似度', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    
    # 计算统计信息
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    std_sim = np.std(similarities)
    
    # 添加统计信息到图中
    stats_text = f"均值: {mean_sim:.4f}\n中位数: {median_sim:.4f}\n标准差: {std_sim:.4f}"
    if threshold is not None:
        stats_text += f"\nOtsu阈值: {threshold:.4f}"
    if knee_threshold is not None:
        stats_text += f"\n拐点阈值: {knee_threshold:.4f}"
    
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 va='top', fontsize=10)
    
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"直方图已保存到: {save_path}")
    
    plt.show()

def calculate_threshold_kneedle(data):
    try:
        from scipy import stats, signal
        
        # 确保数据至少有10个点以进行有意义的KDE(核密度估计)
        if len(data) < 10:
            print("Too few data points for KDE, falling back to Otsu method")
            return threshold_otsu(data)
        
        # 使用KDE估计概率密度函数
        # 自动确定带宽参数，或者可以手动设置
        kde = stats.gaussian_kde(data)
        
        # 在数据范围内创建均匀的评估点
        min_data, max_data = np.min(data), np.max(data)
        x_grid = np.linspace(min_data, max_data, 1000)
        
        # 计算KDE在评估点上的值
        pdf = kde(x_grid)
        
        # 找到局部最小值
        # 使用负的PDF寻找局部最大值，等同于找PDF的局部最小值
        minima_indices = signal.find_peaks(-pdf)[0]
        
        if len(minima_indices) == 0:
            print("No local minima found in KDE, falling back to Otsu method")
            return threshold_otsu(data)
        
        # 从所有局部最小值中选择一个作为阈值
        # 启发式：选择第一个局部最小值，它通常在背景和前景之间
        # 如果想要更复杂的选择逻辑，可以根据具体问题修改
        threshold_value = x_grid[minima_indices[0]]
        
        return threshold_value
    except Exception as e:
        print(f"Error in KDE threshold calculation: {e}, falling back to Otsu method")
        return threshold_otsu(data)

# 方法2: 使用CDF导数分析找拐点
def calculate_threshold_cdf_derivative(data):
    # GMM
    try:
        from sklearn.mixture import GaussianMixture
        
        # 确保数据至少有20个点以拟合GMM
        if len(data) < 20:
            print("Too few data points for GMM, falling back to Otsu method")
            return threshold_otsu(data)
        
        # 将数据重塑为sklearn所需的形状
        data_reshaped = data.reshape(-1, 1)
        
        # 使用2个组件拟合GMM（假设数据来自两个高斯分布）
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(data_reshaped)
        
        # 获取参数
        means = gmm.means_.flatten()  # 均值
        variances = gmm.covariances_.flatten()  # 方差
        weights = gmm.weights_.flatten()  # 权重
        
        # 确保均值以递增顺序排列
        if means[0] > means[1]:
            means = means[::-1]
            variances = variances[::-1]
            weights = weights[::-1]
        
        # 计算两个高斯分布的交叉点
        # 解二次方程: w1*N(x|μ1,σ1²) = w2*N(x|μ2,σ2²)
        # 为了简化，我们使用启发式方法选择两个均值之间的点
        
        # 为了更准确地找到交叉点，我们求解二次方程
        a = 1/(2*variances[1]) - 1/(2*variances[0])
        b = means[0]/variances[0] - means[1]/variances[1]
        c = (means[1]**2)/(2*variances[1]) - (means[0]**2)/(2*variances[0]) - np.log((weights[1]/weights[0]) * np.sqrt(variances[0]/variances[1]))
        
        # 判别式
        discriminant = b**2 - 4*a*c
        
        # 如果有实数解
        if discriminant >= 0 and a != 0:
            # 计算两个解
            threshold1 = (-b + np.sqrt(discriminant)) / (2*a)
            threshold2 = (-b - np.sqrt(discriminant)) / (2*a)
            
            # 选择位于两个均值之间的解
            if means[0] <= threshold1 <= means[1]:
                return threshold1
            elif means[0] <= threshold2 <= means[1]:
                return threshold2
        
        # 如果没有找到适当的解，使用两个均值的加权平均
        threshold_value = (means[0] * weights[0] + means[1] * weights[1]) / (weights[0] + weights[1])
        
        return threshold_value
    except Exception as e:
        print(f"Error in GMM threshold calculation: {e}, falling back to Otsu method")
        return threshold_otsu(data)
def main():
    # 定义注意力权重文件夹路径
    attn_weights_dir = "./attn_weights"
    
    # 计算相似度
    print(f"从{attn_weights_dir}加载.pt文件...")
    similarities, block_step_info = calculate_cosine_similarity(attn_weights_dir)
    
    print(f"共处理了{len(similarities)}个文件")
    print(f"相似度 - 最小值: {min(similarities):.4f}, 最大值: {max(similarities):.4f}")
    
    # 应用Otsu阈值
    otsu_threshold, otsu_above_threshold = apply_otsu_threshold(similarities, block_step_info)
    print(f"\nOtsu阈值: {otsu_threshold:.4f}")
    print(f"发现{len(otsu_above_threshold)}个高于Otsu阈值的块-时间步组合")
    
    # 应用拐点检测阈值
    knee_threshold, knee_above_threshold = apply_knee_threshold(similarities, block_step_info)
    print(f"\n拐点阈值: {knee_threshold:.4f}")
    print(f"发现{len(knee_above_threshold)}个高于拐点阈值的块-时间步组合")
    
    # 输出高于Otsu阈值的组合
    print("\n高于Otsu阈值的前10个块-时间步组合（按相似度降序）:")
    print("{:<10} {:<10} {:<15} {:<}".format("块ID", "步骤ID", "余弦相似度", "文件"))
    for i, (block_id, step_id, sim, file_path) in enumerate(otsu_above_threshold[:10]):
        print("{:<10} {:<10} {:<15.4f} {:<}".format(block_id, step_id, sim, os.path.basename(file_path)))
    
    # 输出高于拐点阈值的组合
    print("\n高于拐点阈值的前10个块-时间步组合（按相似度降序）:")
    print("{:<10} {:<10} {:<15} {:<}".format("块ID", "步骤ID", "余弦相似度", "文件"))
    for i, (block_id, step_id, sim, file_path) in enumerate(knee_above_threshold[:10]):
        print("{:<10} {:<10} {:<15.4f} {:<}".format(block_id, step_id, sim, os.path.basename(file_path)))
    
    # 将所有高于阈值的组合写入文件
    # Otsu阈值结果
    otsu_output_file = "high_sim_otsu_combinations.txt"
    with open(otsu_output_file, "w") as f:
        f.write("{:<10} {:<10} {:<15} {:<}\n".format("块ID", "步骤ID", "余弦相似度", "文件"))
        for block_id, step_id, sim, file_path in otsu_above_threshold:
            f.write("{:<10} {:<10} {:<15.4f} {:<}\n".format(block_id, step_id, sim, os.path.basename(file_path)))
    print(f"\n所有高于Otsu阈值的组合已保存到: {otsu_output_file}")
    
    # 拐点阈值结果
    knee_output_file = "high_sim_knee_combinations.txt"
    with open(knee_output_file, "w") as f:
        f.write("{:<10} {:<10} {:<15} {:<}\n".format("块ID", "步骤ID", "余弦相似度", "文件"))
        for block_id, step_id, sim, file_path in knee_above_threshold:
            f.write("{:<10} {:<10} {:<15.4f} {:<}\n".format(block_id, step_id, sim, os.path.basename(file_path)))
    print(f"所有高于拐点阈值的组合已保存到: {knee_output_file}")
    
    # 绘制直方图
    save_path = "./attention_similarity.png"
    plot_similarity_histogram(similarities, otsu_threshold, knee_threshold, save_path)

if __name__ == "__main__":
    main()
