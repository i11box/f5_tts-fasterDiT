import torch
import os

# 设置文件路径
# file_path = "step6_attn_weights_before_softmax.pt"
file_path = "step7_key.pt"

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # 加载 .pt 文件
    data = torch.load(file_path)

    # 打印数据的形状
    print(f"Data shape: {data.shape}")

    # 打印部分值
    print("First 5x5 values:")
    print(data[0, :5, :5])

    print("\nLast 5x5 values:")
    print(data[0, -5:, -5:])

    print("\nMiddle 5x5 values:")
    mid = data.shape[1] // 2
    print(data[0, mid-2:mid+3, mid-2:mid+3])
