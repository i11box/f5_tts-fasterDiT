import matplotlib.pyplot as plt
import re
import os
import numpy as np

# 文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'diagonal_matches_results.txt')

# 尝试不同的编码方式读取文件
encodings = ['utf-8', 'gbk', 'utf-16', 'latin-1']
similarities = []

for encoding in encodings:
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
            
        # 提取相似度值
        pattern = r'相似度: ([0-9.]+)'
        for line in lines:
            match = re.search(pattern, line)
            if match:
                similarity = float(match.group(1))
                similarities.append(similarity)
        
        # 如果成功读取数据，跳出循环
        if similarities:
            print(f'成功使用 {encoding} 编码读取文件')
            print(f'读取到 {len(similarities)} 个相似度值')
            break
            
    except UnicodeDecodeError:
        continue
    except Exception as e:
        print(f'使用 {encoding} 编码读取时出错: {e}')
        continue

if not similarities:
    print('无法读取文件或提取相似度值')
    exit(1)

# 创建直方图
plt.figure(figsize=(10, 6))

# 使用更细致的分组
# 创建更精细的bins，范围从最小值到最大值，共50个分组
min_val = min(similarities)
max_val = max(similarities)
bins = np.linspace(min_val, max_val, 30)

n, bins, patches = plt.hist(similarities, bins=bins, edgecolor='black', alpha=0.7)

# 设置图表标题和标签
plt.title('相似度分布直方图')
plt.xlabel('相似度')
plt.ylabel('频数')
plt.grid(axis='y', alpha=0.75)

# 美化图表
plt.tight_layout()

# 保存图表
output_path = os.path.join(os.path.dirname(file_path), 'similarity_histogram.png')
plt.savefig(output_path)
print(f'直方图已保存至: {output_path}')

# 显示图表
plt.show()
