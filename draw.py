import matplotlib.pyplot as plt
import numpy as np

# 数据
thresholds = ['No Compression', '0.1', '0.15', '0.2']

# 10字数据
flops_10_total = [3191.88, 2730.01, 2279.08, 1876.59]
flops_10_attn = [1272.47, 1066.72, 885.97, 725.51]
flops_10_linear = [1919.41, 1663.30, 1393.12, 1151.08]

# 70字数据
flops_70_total = [16997.54, 14410.96, 12003.86, 9860.47]
flops_70_attn = [11282.23, 9458.25, 7855.65, 6432.96]
flops_70_linear = [5715.31, 4952.71, 4148.21, 3427.51]

# 设置图形大小
plt.figure(figsize=(15, 6))

# 设置柱状图的宽度和位置
bar_width = 0.25
r1 = np.arange(len(thresholds))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# 创建子图
plt.subplot(1, 2, 1)
plt.bar(r1, flops_10_total, width=bar_width, label='Total', color='skyblue')
plt.bar(r2, flops_10_attn, width=bar_width, label='Attention', color='lightgreen')
plt.bar(r3, flops_10_linear, width=bar_width, label='Linear', color='salmon')

plt.xlabel('Compression Threshold')
plt.ylabel('GFLOPS')
plt.title('FLOPS Distribution (10 Characters)')
plt.xticks([r + bar_width for r in range(len(thresholds))], thresholds)
plt.legend()

# 添加数值标签
for i, v in enumerate(flops_10_total):
    plt.text(r1[i], v, f'{v:.1f}', ha='center', va='bottom',fontsize = 8)
for i, v in enumerate(flops_10_attn):
    plt.text(r2[i], v, f'{v:.1f}', ha='center', va='bottom',fontsize = 8)
for i, v in enumerate(flops_10_linear):
    plt.text(r3[i], v, f'{v:.1f}', ha='center', va='bottom',fontsize = 8)

plt.subplot(1, 2, 2)
plt.bar(r1, flops_70_total, width=bar_width, label='Total', color='skyblue')
plt.bar(r2, flops_70_attn, width=bar_width, label='Attention', color='lightgreen')
plt.bar(r3, flops_70_linear, width=bar_width, label='Linear', color='salmon')

plt.xlabel('Compression Threshold')
plt.ylabel('GFLOPS')
plt.title('FLOPS Distribution (70 Characters)')
plt.xticks([r + bar_width for r in range(len(thresholds))], thresholds)
plt.legend()

# 添加数值标签
for i, v in enumerate(flops_70_total):
    plt.text(r1[i], v, f'{v:.1f}', ha='center', va='bottom',fontsize = 8)
for i, v in enumerate(flops_70_attn):
    plt.text(r2[i], v, f'{v:.1f}', ha='center', va='bottom',fontsize = 8)
for i, v in enumerate(flops_70_linear):
    plt.text(r3[i], v, f'{v:.1f}', ha='center', va='bottom',fontsize = 8)

# 调整布局并保存
plt.tight_layout()
plt.show()
plt.close()