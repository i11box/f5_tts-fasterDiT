import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
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

# 生成对角线模板图像
def generate_diagonal_template(size=(256, 256), add_anti_diagonal=True):
    """生成一个对角线模板图像
    
    参数:
        size: 图像尺寸 (高度, 宽度)
        add_anti_diagonal: 是否添加副对角线
        
    返回:
        对角线模板图像
    """
    # 创建空白图像(黑色背景)
    template = np.zeros(size, dtype=np.uint8)
    h, w = size
    
    # 添加主对角线(左上到右下)
    for i in range(min(h, w)):
        # 使用线性插值计算对角线上的像素坐标
        row = int(i * h / min(h, w)) if h > w else i
        col = int(i * w / min(h, w)) if w > h else i
        template[row, col] = 255
    
    return template

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
    cv2.line(diag_mask, (0, 0), (w, h), 255, max(h, w) // 10)  # 主对角线
    
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
def compute_similarity(template_img, img):
    # 使用SSIM计算结构相似度
    # 注意：SSIM需要直接比较图像，因此需要调整大小以匹配模板
    if template_img.shape != img.shape:
        img = cv2.resize(img, (template_img.shape[1], template_img.shape[0]))
    
    # 计算SSIM指标
    similarity = ssim(template_img, img, data_range=255)
    return similarity

# 4. 可视化结果
def visualize_matches(template_img, matches, similarities, is_generated_template=False):
    plt.figure(figsize=(15, 10))
    
    # 显示模板
    plt.subplot(1, min(6, len(matches)+1), 1)
    if is_generated_template:
        # 直接显示生成的模板图像
        plt.imshow(template_img, cmap='gray')
        plt.title('生成的对角线模板')
    else:
        # 从文件中读取模板
        template_display = cv2.imread(template_img)
        template_display = cv2.cvtColor(template_display, cv2.COLOR_BGR2RGB)
        plt.imshow(template_display)
        plt.title('热力图模板')
    plt.axis('off')
    
    # 显示匹配结果（最多5个）
    for i, (match_path, sim) in enumerate(zip(matches[:5], similarities[:5])):
        plt.subplot(1, min(6, len(matches)+1), i+2)
        match_img = cv2.imread(match_path)
        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        plt.imshow(match_img)
        plt.title(f'相似度: {sim:.4f}')
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
def template_match(template_img, similarity_threshold=0.7, use_generated_template=False):
    try:
        # 提取模板特征（我们直接使用图像而不是特征）
        if not use_generated_template:
            # 如果使用真实热力图，需要预处理
            template_img = preprocess_heatmap(template_img)
        _, template_lines = extract_diagonal_features(template_img)
        
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
                    # 计算SSIM相似度
                    similarity = compute_similarity(template_img, img)
                    
                    # 如果相似度超过阈值，添加到匹配结果
                    if similarity >= similarity_threshold:
                        matches.append(img_path)
                        similarities.append(similarity)
                        match_indices.append((block, step))
                        print(f"匹配: Block {block}, Step {step}, 相似度(SSIM) {similarity:.4f}")
                except Exception as e:
                    print(f"处理图像 {block}/{step} 时出错: {e}")
                    continue
        
        # 按相似度排序
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_matches = [matches[i] for i in sorted_indices]
        sorted_similarities = [similarities[i] for i in sorted_indices]
        sorted_match_indices = [match_indices[i] for i in sorted_indices]
        
        # 可视化匹配结果
        if sorted_matches:
            visualize_matches(template_img if use_generated_template else sorted_matches[0], 
                         sorted_matches, sorted_similarities, is_generated_template=use_generated_template)
        else:
            print("未找到匹配的热力图")
        
        return sorted_match_indices, sorted_similarities
    
    except Exception as e:
        print(f"模板匹配过程中发生错误: {e}")
        return [], []

# 示例使用
if __name__ == "__main__":
    print("请选择使用方式：")
    print("1. 使用现有热力图作为模板")
    print("2. 使用生成的对角线模板")
    
    try:
        choice = int(input())
        
        # 设定相似度阈值
        threshold = 0.9
        
        if choice == 1:
            # 使用现有热力图作为模板
            print("请输入模板热力图的block和step (例如: 5 10):")
            template_block, template_step = map(int, input().split())
            print(f"使用Block {template_block}, Step {template_step}作为模板，阈值设为{threshold}")
            
            # 执行模板匹配
            template_path = generate_image_paths(template_block, template_step)
            matches, similarities = template_match(template_path, threshold, use_generated_template=False)
        
        elif choice == 2:
            # 使用生成的对角线模板
            print("生成对角线模板...")
            print("是否包含副对角线？(y/n, 默认n):")
            include_anti_diagonal = input().lower() == 'y'
            
            # 获取第一张图像的尺寸，用于创建合适大小的模板
            sample_img = preprocess_heatmap(generate_image_paths(0, 0))
            h, w = sample_img.shape
            
            # 生成对角线模板
            template = generate_diagonal_template(size=(h, w), add_anti_diagonal=include_anti_diagonal)
            
            # 保存生成的模板
            cv2.imwrite('generated_template.png', template)
            print(f"生成的对角线模板已保存至 generated_template.png")
            print(f"使用生成的对角线模板，阈值设为{threshold}")
            
            # 执行模板匹配
            matches, similarities = template_match(template, threshold, use_generated_template=True)
        
        else:
            print("无效的选择，请选择1或2")
            exit(1)
        
        # 打印结果
        print(f"\n找到 {len(matches)} 个匹配结果:")
        for (block, step), sim in zip(matches[:10], similarities[:10]):
            print(f"Block {block}, Step {step}, 相似度: {sim:.4f}")
        
        # 保存匹配结果到文件
        with open('diagonal_matches_results.txt', 'w') as f:
            if choice == 1:
                f.write(f"模板: Block {template_block}, Step {template_step}\n")
            else:
                f.write(f"使用生成的对角线模板\n")
            f.write(f"阈值: {threshold}\n")
            f.write(f"找到 {len(matches)} 个匹配结果:\n")
            for (block, step), sim in zip(matches, similarities):
                f.write(f"Block {block}, Step {step}, 相似度: {sim:.4f}\n")
        print("\n结果已保存至 diagonal_matches_results.txt")
        
    except ValueError as e:
        print(f"输入错误: {e}")
        print("请重新启动脚本并按提示输入")
