import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from pathlib import Path

# 基础路径
BASE_PATH = 'assets/attention_heatmaps/by_block'

# 1. 预处理：裁剪热力图主体区域
def preprocess_heatmap(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自动裁剪（假设热力图位于图像中心区域）
    h, w = gray.shape
    cropped = gray[h//4:3*h//4, w//4:3*w//4]  # 调整裁剪比例
    return cropped

# 2. 对角线特征提取
def extract_diagonal_features(img):
    # 1. 边缘检测
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    
    # 2. 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                          minLineLength=img.shape[0]//4, maxLineGap=20)
    
    # 创建直线蒙版图像
    line_mask = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    
    # 3. 提取HOG特征（小型特征提取器）
    hog_features = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False)
    
    # 4. 生成对角线热图并作为特征之一
    diag_mask = np.zeros_like(img)
    h, w = img.shape
    thickness = max(h, w) // 10  # 对角线粗细
    cv2.line(diag_mask, (0, 0), (w, h), 255, thickness)  # 主对角线
    cv2.line(diag_mask, (0, h), (w, 0), 255, thickness)  # 副对角线
    
    # 计算与对角线模板的相似度
    diag_similarity = np.sum(diag_mask & img) / np.sum(diag_mask)
    line_similarity = np.sum(line_mask & img) / (np.sum(line_mask) + 1e-10)
    
    # 5. 统计特征
    std_dev = np.std(img)
    grad_y, grad_x = np.gradient(img)
    gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    mean_gradient = np.mean(gradient_magnitude)
    
    # 组合特征
    combined_features = np.concatenate([
        hog_features, 
        [diag_similarity, line_similarity, std_dev, mean_gradient]
    ])
    
    return combined_features, line_mask

# 3. 计算与模板的相似度
def compute_similarity(template_features, image_features):
    # 使用余弦相似度
    sim = cosine_similarity(template_features.reshape(1, -1), 
                           image_features.reshape(1, -1))[0][0]
    return sim

# 4. 可视化结果
def visualize_matches(template_path, matches, similarities):
    plt.figure(figsize=(15, 10))
    
    # 显示模板
    plt.subplot(1, min(6, len(matches)+1), 1)
    template_img = cv2.imread(template_path)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
    plt.imshow(template_img)
    plt.title('Template')
    plt.axis('off')
    
    # 显示匹配结果（最多5个）
    for i, (match_path, sim) in enumerate(zip(matches[:5], similarities[:5])):
        plt.subplot(1, min(6, len(matches)+1), i+2)
        match_img = cv2.imread(match_path)
        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        plt.imshow(match_img)
        plt.title(f'Sim: {sim:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('diagonal_matches.png')
    plt.close()
    print(f"结果已保存至 diagonal_matches.png")

# 5. 生成图像路径
def generate_image_paths(block, step):
    img_path = f"{block}/conditional/step_{step}.png"
    return os.path.join(BASE_PATH, img_path)

# 6. 主函数：模板匹配
def template_match(template_block, template_step, similarity_threshold=0.7):
    # 获取模板图像
    template_path = generate_image_paths(template_block, template_step)
    
    try:
        # 预处理模板图像
        template_img = preprocess_heatmap(template_path)
        # 提取模板特征
        template_features, template_lines = extract_diagonal_features(template_img)
        
        # 可视化模板检测的线
        cv2.imwrite('template_lines.png', template_lines)
        print(f"模板线检测结果已保存至 template_lines.png")
        
        # 存储匹配结果
        matches = []
        similarities = []
        match_indices = []
        
        # 遍历所有热力图进行匹配
        for block in range(22):
            for step in range(31):
                try:
                    # 构建图像路径
                    img_path = generate_image_paths(block, step)
                    # 预处理图像
                    img = preprocess_heatmap(img_path)
                    # 提取特征
                    img_features, _ = extract_diagonal_features(img)
                    # 计算相似度
                    similarity = compute_similarity(template_features, img_features)
                    
                    # 如果相似度超过阈值，添加到匹配结果
                    if similarity >= similarity_threshold:
                        matches.append(img_path)
                        similarities.append(similarity)
                        match_indices.append((block, step))
                        print(f"匹配: Block {block}, Step {step}, 相似度 {similarity:.4f}")
                except Exception as e:
                    print(f"处理图像 {block}/{step} 时出错: {e}")
                    continue
        
        # 按相似度排序
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_matches = [matches[i] for i in sorted_indices]
        sorted_similarities = [similarities[i] for i in sorted_indices]
        sorted_match_indices = [match_indices[i] for i in sorted_indices]
        
        # # 使用 Otsu 方法计算阈值
        # threshold = threshold_otsu(np.array(similarities))
        # print(f"Otsu Threshold: {threshold}")
        
        # # 划分数据
        # below_threshold = sorted_similarities[sorted_similarities < threshold]
        # above_threshold = sorted_similarities[sorted_similarities >= threshold]
        
        return sorted_match_indices, sorted_similarities
    
    except Exception as e:
        print(f"模板匹配过程中发生错误: {e}")
        return [], []

# 示例使用
if __name__ == "__main__":
    # 让用户选择一个模板
    print("请输入模板热力图的block和step (例如: 5 10):")
    try:
        template_block, template_step = map(int, input().split())
        # 设定相似度阈值
        threshold = 0.85
        print(f"使用Block {template_block}, Step {template_step}作为模板，阈值设为{threshold}")
        
        # 执行模板匹配
        matches, similarities = template_match(template_block, template_step, threshold)
        
        # 打印结果
        print(f"\n找到 {len(matches)} 个匹配结果:")
        for (block, step), sim in zip(matches[:10], similarities[:10]):
            print(f"Block {block}, Step {step}, 相似度: {sim:.4f}")
        
        # 保存匹配结果到文件
        with open('diagonal_matches_results.txt', 'w') as f:
            f.write(f"模板: Block {template_block}, Step {template_step}\n")
            f.write(f"阈值: {threshold}\n")
            f.write(f"找到 {len(matches)} 个匹配结果:\n")
            for (block, step), sim in zip(matches, similarities):
                f.write(f"Block {block}, Step {step}, 相似度: {sim:.4f}\n")
        print("\n结果已保存至 diagonal_matches_results.txt")
        
    except ValueError:
        print("输入格式错误，请输入两个整数，用空格分隔")
