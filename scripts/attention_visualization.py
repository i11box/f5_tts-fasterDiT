import torch
import matplotlib.pyplot as plt
import numpy as np
from flash_attn import flash_attn_func

def get_device():
    """获取可用的设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        raise RuntimeError("Flash attention requires CUDA. Please run this script on a GPU.")

def generate_sample_data(batch_size=1, seq_len=32, dim=64, device=None, dtype=None):
    """生成示例的Q,K,V数据，匹配原始维度 [batch_size, seq_len, dim]"""
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = torch.float16  # flash attention要求fp16
    
    query = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    return query, key, value

def sampled_attention(query, key, value, sample_ratio=0.25, window_size=None, mode='window'):
    """采样注意力计算
    
    Args:
        query: [batch_size, seq_len, dim]
        key: [batch_size, seq_len, dim] 
        value: [batch_size, seq_len, dim]
        sample_ratio: 采样比例
        window_size: 窗口大小,如果提供则在窗口内采样
        mode: 采样模式，'window'或'uniform'或'dialected'
    """
    device = query.device
    dtype = query.dtype
    batch_size, seq_len, dim = query.shape
    
    if window_size is not None and mode == 'window':
        # 创建窗口mask
        window_mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            window_mask[i, start:end] = 1
        
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / (dim ** 0.5)
        
        # 应用窗口mask
        scores = scores * window_mask
        scores = scores.masked_fill(window_mask == 0, float('-inf'))
        
        # 计算注意力权重
        weights = torch.softmax(scores, dim=-1)
        weights = weights.to(dtype=dtype)
        
        # 计算输出
        output = weights
    elif mode == 'uniform':
        # 均匀采样
        sampled_size = int(seq_len * sample_ratio)
        stride = max(1, seq_len // sampled_size)  # 计算步长，至少为1

        # 生成均匀分布的索引
        indices = (torch.arange(sampled_size, device=device) * stride)
        indices = torch.clamp(indices, max=seq_len-1).long()  # 限制索引范围

        # 稀疏注意力计算
        sparse_query = query[:, indices]
        sparse_key = key[:, indices]
        sparse_value = value[:, indices]
        
        # 计算注意力权重
        sparse_scores = torch.matmul(sparse_query, sparse_key.transpose(-2, -1)) / (dim ** 0.5)
        sparse_weights = torch.softmax(sparse_scores, dim=-1)
        
        # 计算输出
        sparse_output = torch.matmul(sparse_weights, sparse_value)
        
        # 映射回原始序列
        output = torch.zeros_like(query)
        output[:, indices] = sparse_output
        
    elif mode == 'dialected':
        # 计算膨胀步长
        dilation = max(1, int(1 / sample_ratio))
        
        # 生成dilated采样位置
        base_indices = torch.arange(0, seq_len, dilation, device=query.device)
        
        # 为每个采样位置构建局部窗口
        all_indices = []
        for idx in base_indices:
            # 计算窗口范围
            window_start = max(0, idx - window_size)
            window_end = min(seq_len, idx + window_size + 1)
            window_indices = torch.arange(window_start, window_end, device=query.device)
            all_indices.append(window_indices)
        
        # 合并所有索引并去重
        indices = torch.unique(torch.cat(all_indices))
        sparse_query = query[:, indices]
        sparse_key = key[:, indices]
        sparse_value = value[:, indices]
        
        # 计算稀疏注意力权重
        sparse_scores = torch.matmul(sparse_query, sparse_key.transpose(-2, -1)) / (dim ** 0.5)
        sparse_weights = torch.softmax(sparse_scores, dim=-1)
        
        # 计算输出
        sparse_output = torch.matmul(sparse_weights, sparse_value)
        
        # 映射回原始序列
        output = torch.zeros_like(query)
        output[:, indices] = sparse_output
    elif mode == 'window + random':
        # 1. 计算窗口注意力
        w_size = window_size
        
        # 使用 flash_attn_func 的窗口注意力
        window_output = flash_attn_func(
            query, key, value,
            causal=False,
            window_size=(-w_size, w_size)
        )
        
        # 2. 计算随机注意力
        random_size = int(seq_len * sample_ratio)
        
        # 随机采样索引
        random_indices = torch.stack(
            [torch.randperm(seq_len, device=device)[:random_size] for _ in range(batch_size)]
        )  # [batch_size, random_size]
        
        # 切片获取随机采样的 query、key 和 value
        random_query = torch.stack([query[b, idx] for b, idx in enumerate(random_indices)])  # [batch_size, random_size, heads, head_dim]
        random_key = torch.stack([key[b, idx] for b, idx in enumerate(random_indices)])
        random_value = torch.stack([value[b, idx] for b, idx in enumerate(random_indices)])
        
        # 使用 flash_attn_func 计算随机注意力
        random_output = flash_attn_func(
            random_query, random_key, random_value,
            causal=False
        )  # [batch_size, random_size, heads, head_dim]
        
        # 将随机注意力的结果映射回原始序列长度
        random_output_full = torch.zeros_like(query, dtype=dtype, device=device)  # [batch_size, seq_len, heads, head_dim]
        for b, idx in enumerate(random_indices):
            random_output_full[b, idx] = random_output[b]
        
        # 3. 合并两种注意力
        # 创建窗口有效掩码：1 表示窗口注意力有效，0 表示可以使用随机注意力
        window_mask = torch.zeros((seq_len, seq_len), device=device)
        for i in range(seq_len):
            start = max(0, i - w_size)
            end = min(seq_len, i + w_size + 1)
            window_mask[i, start:end] = 1
        
        num_heads = 8
        
        window_valid_mask = window_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len)  # [batch_size, heads, seq_len, seq_len]
        
        # 合并权重
        merged_output = torch.where(
            window_valid_mask.bool(),  # 条件：窗口注意力是否有效
            window_output,             # 窗口注意力结果
            random_output_full         # 随机注意力结果
        )
        
        return merged_output
    
    return output

def visualize_attention_patterns(seq_len=32, dim=64, window_size=4):
    """可视化不同注意力模式的效果"""
    device = 'cuda'
    dtype = torch.float16
    
    # 生成示例数据
    query, key, value = generate_sample_data(1, seq_len, dim, device, dtype)
    
    # 计算不同模式的注意力
    dialected_output = sampled_attention(query, key, value, sample_ratio=0.125, window_size=window_size, mode='window + random')
    
    # 可视化输出激活值的热力图
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    im = axes.imshow(dialected_output[0, :, :].detach().cpu().float().numpy(), cmap='viridis')
    axes.set_title(f'Dialected')
    plt.colorbar(im, ax=axes)
    
    plt.tight_layout()
    # plt.savefig('attention_patterns.png')
    plt.show()

if __name__ == '__main__':
    visualize_attention_patterns()
