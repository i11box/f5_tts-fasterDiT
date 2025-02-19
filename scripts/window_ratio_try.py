import json
import glob
import matplotlib.pyplot as plt
import numpy as np

def count_methods(methods_list):
    # 将二维列表展平并统计每种方法的出现次数
    flat_list = [item for sublist in methods_list for item in sublist]
    counts = {
        'full_attention': flat_list.count('full_attention'),
        'wars': flat_list.count('wars'),
        'AST': flat_list.count('AST'),
        'ASC': flat_list.count('ASC'),
        'wars+ASC': flat_list.count('wars+ASC')
    }
    return counts

def analyze_ratios():
    # 需要分析的ratio列表
    ratios = ['0.06', '0.0625', '0.07', '0.08']
    results = {}
    
    # 读取并统计每个ratio对应的数据
    for ratio in ratios:
        filename = f'data/methods/32_0.2_{ratio}.json'
        with open(filename, 'r') as f:
            data = json.load(f)
            results[ratio] = count_methods(data['methods'])
    
    # 绘制柱状图
    methods = ['full_attention', 'wars', 'AST', 'ASC', 'wars+ASC']
    x = np.arange(len(ratios))
    width = 0.15
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        counts = [results[ratio][method] for ratio in ratios]
        ax.bar(x + i*width, counts, width, label=method)
    
    ax.set_xlabel('窗口比例(ratio)')
    ax.set_ylabel('使用次数')
    ax.set_title('不同窗口比例下各方法的使用分布')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(ratios)
    ax.legend()
    
    plt.savefig('方法分布.png')
    plt.close()
    
    # 打印数值结果
    print("\n各ratio下的方法使用次数：")
    for ratio in ratios:
        print(f"\nratio = {ratio}:")
        for method, count in results[ratio].items():
            print(f"{method}: {count}")
            
    # 分析趋势并建议下一步尝试的ratio
    return suggest_next_ratios(results)

def suggest_next_ratios(results):
    # 将ratio转换为浮点数并排序
    ratios = sorted([float(r) for r in results.keys()])
    
    # 寻找方法分布发生显著变化的区域
    changes = []
    for i in range(len(ratios)-1):
        r1, r2 = str(ratios[i]), str(ratios[i+1])
        total_change = sum(abs(results[r2][m] - results[r1][m]) for m in results[r1])
        changes.append((ratios[i], ratios[i+1], total_change))
    
    # 找出ratio序列中的间隔
    gaps = [(ratios[i], ratios[i+1]) for i in range(len(ratios)-1)]
    largest_gaps = sorted(gaps, key=lambda x: x[1]-x[0], reverse=True)[:2]
    
    # 生成建议的新ratio值
    suggestions = []
    
    # 添加最大间隔的中点
    for gap_start, gap_end in largest_gaps:
        suggestions.append((gap_end + gap_start) / 2)
        
    return sorted(suggestions)

if __name__ == "__main__":
    next_ratios = analyze_ratios()
    print("\n建议下一步尝试的ratio值:", next_ratios)