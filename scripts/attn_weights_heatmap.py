import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体为 SimHei，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_attention_heatmap(attention, save_path, title="Attention Heatmap", device="cpu"):
    """
    绘制注意力热力图并保存到指定路径
    """
    # 将张量移动到 CPU 并转换为 NumPy 数组
    attention = attention.squeeze(0).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap='binary_r',vmin = 0,vmax = 0.5)
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig(save_path)  # 保存图像
    plt.close()  # 关闭图像以释放内存

def process_attention_weights(input_dir, output_dir, device="cpu"):
    """
    处理 attn_weights 文件夹中的所有注意力权重文件
    """
    # 创建输出文件夹
    conditional_output_dir = os.path.join(output_dir, "cond")
    unconditional_output_dir = os.path.join(output_dir, "uncond")
    os.makedirs(conditional_output_dir, exist_ok=True)
    os.makedirs(unconditional_output_dir, exist_ok=True)

    # 遍历 block_id 和 step
    num_blocks = 22
    num_steps = 31

    for block_id in range(num_blocks):
        for step in range(num_steps):
            # 构造文件名
            file_name = f"block_{block_id}_step_{step}.pt"
            file_path = os.path.join(input_dir, file_name)

            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            # 加载注意力权重到指定设备
            weights = torch.load(file_path, map_location=device)

            # 分离条件输出和无条件输出
            conditional_weights = weights[0].to(device)  # 第一层是条件输出
            unconditional_weights = weights[1].to(device)  # 第二层是无条件输出

            # 创建按block和按step分组的目录
            by_block_cond_dir = os.path.join(output_dir, 'by_block', str(block_id), 'conditional')
            by_block_uncond_dir = os.path.join(output_dir, 'by_block', str(block_id), 'unconditional')
            by_step_cond_dir = os.path.join(output_dir, 'by_step', str(step), 'conditional')
            by_step_uncond_dir = os.path.join(output_dir, 'by_step', str(step), 'unconditional')

            # 确保目录存在
            os.makedirs(by_block_cond_dir, exist_ok=True)
            os.makedirs(by_block_uncond_dir, exist_ok=True)
            os.makedirs(by_step_cond_dir, exist_ok=True)
            os.makedirs(by_step_uncond_dir, exist_ok=True)

            # 2. 按block分组的保存路径
            by_block_cond_save_path = os.path.join(
                by_block_cond_dir,
                f"step_{step}.png"
            )
            by_block_uncond_save_path = os.path.join(
                by_block_uncond_dir,
                f"step_{step}.png"
            )

            # 3. 按step分组的保存路径
            by_step_cond_save_path = os.path.join(
                by_step_cond_dir,
                f"block_{block_id}.png"
            )
            by_step_uncond_save_path = os.path.join(
                by_step_uncond_dir,
                f"block_{block_id}.png"
            )

            # 绘制并保存条件输出热力图（按block分组）
            plot_attention_heatmap(
                conditional_weights,
                by_block_cond_save_path,
                title=f"Block {block_id} Step {step} Conditional Attention",
                device=device
            )

            # 绘制并保存无条件输出热力图（按block分组）
            plot_attention_heatmap(
                unconditional_weights,
                by_block_uncond_save_path,
                title=f"Block {block_id} Step {step} Unconditional Attention",
                device=device
            )

            # 绘制并保存条件输出热力图（按step分组）
            plot_attention_heatmap(
                conditional_weights,
                by_step_cond_save_path,
                title=f"Block {block_id} Step {step} Conditional Attention",
                device=device
            )

            # 绘制并保存无条件输出热力图（按step分组）
            plot_attention_heatmap(
                unconditional_weights,
                by_step_uncond_save_path,
                title=f"Block {block_id} Step {step} Unconditional Attention",
                device=device
            )

            print(f"Processed: {file_name}")

            # 清理 GPU 缓存
            if device.type == "cuda":
                torch.cuda.empty_cache()

# 主函数
if __name__ == "__main__":
    input_dir = "./attn_weights"  # 输入文件夹路径
    output_dir = "./assets/attention_heatmaps"  # 输出文件夹路径

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 处理注意力权重
    process_attention_weights(input_dir, output_dir, device=device)
    # attention = torch.load("step6_attn_weights_after_softmax.pt")
    # attention_cond,attention_uncond = attention.chunk(2,dim=0)
    # plot_attention_heatmap(attention_cond,'')
    # plot_attention_heatmap(attention_uncond,'')