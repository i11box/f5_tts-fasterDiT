import torch
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_attention_heatmap(attention, title="Average Attention Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention.squeeze(0).cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()

weights = torch.load('./attention_weights/attn_weights_block_None.pt')
average_attention = weights.mean(dim=1)

# 绘制平均注意力热力图
plot_attention_heatmap(average_attention, title="Average Attention Heatmap")

print(weights.shape)

