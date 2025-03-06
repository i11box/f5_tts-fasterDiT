import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.measure import block_reduce

# 1. 预处理：裁剪热力图主体区域
def preprocess_heatmap(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自动裁剪（假设热力图位于图像中心区域）
    h, w = gray.shape
    cropped = gray[h//4:3*h//4, w//4:3*w//4]  # 调整裁剪比例
    return cropped

def extract_matrix_features(cropped_img):
    # 下采样减少计算量
    resized = block_reduce(cropped_img, block_size=(2,2), func=np.mean)
    
    # 获取y和x方向的梯度
    grad_y, grad_x = np.gradient(resized)
    
    # 计算梯度幅度
    gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    
    # 提取统计特征
    features = [
        np.mean(resized), np.std(resized),
        np.max(resized), np.min(resized),
        np.mean(gradient_magnitude)  # 梯度幅度特征
    ]
    return features

# 3. 聚类
def cluster_heatmaps(n_clusters=2):
    features = []
    valid_paths = []
    index = []
    
    # 1. 收集有效特征和路径
    for block in range(22):
        for step in range(31):
            try:
                path = generate_image_paths(block, step)
                cropped = preprocess_heatmap(path)
                feat = extract_matrix_features(cropped)
                
                # 检查是否有NaN值
                if not np.any(np.isnan(feat)):
                    features.append(feat)
                    valid_paths.append(path)
                    index.append((block, step))
            except Exception as e:
                print(f"处理图像 {path} 时出错: {e}")
    
    # 转换为NumPy数组
    features = np.array(features)
    if len(features) == 0:
        print("错误: 没有有效的特征可供处理")
        return [-1] * len(valid_paths)  # 返回默认标签
    
    # 2. 安全地标准化特征
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    min_nonzero_std = np.min(stds[stds > 0]) if np.any(stds > 0) else 1.0
    
    # 防止除以零: 如果标准差为0，将其设为1
    stds[stds < 1e-10] = min_nonzero_std
    
    # 标准化特征
    features = (features - means) / stds
    
    # 3. 处理可能剩余的NaN值
    if np.any(np.isnan(features)):
        print("警告: 标准化后仍有NaN值，已替换为0")
        features = np.nan_to_num(features)
    
    # 4. PCA降维 - 确保组件数量合理
    n_components = min(4, features.shape[1], features.shape[0] - 1)
    if n_components <= 0:
        print("错误: 有效特征数量不足，无法进行PCA降维")
        return [-1] * len(valid_paths)
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    
    # 5. K-Means聚类 - 确保聚类数量合理
    n_clusters = min(n_clusters, len(features))
    if n_clusters < 1:
        n_clusters = 1
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced)
    
    # 6. 返回与原始路径对应的标签
    path_to_label = {index: label for index, label in zip(index, labels)}
    lable_0 = [(index, path_to_label.get(index, -1)) for index in index if path_to_label.get(index, -1) == 0]
    label_1 = [(index, path_to_label.get(index, -1)) for index in index if path_to_label.get(index, -1) == 1]
    return lable_0, label_1


# 产生所有图片路径
def generate_image_paths(block, step):
    base_path = 'assets/attention_heatmaps/by_block'
    img_path = f"{block}/conditional/step_{step}.png"
    return os.path.join(base_path, img_path)

label_0,label_1 = cluster_heatmaps(n_clusters=2)

print(f"label_0共{len(label_0)}个")
for label in label_0:
    print(f"{label[0]}: {label[1]}")

print(f"label_1共{len(label_1)}个")
for label in label_1:
    print(f"{label[0]}: {label[1]}")


