import re
import json
import os

def parse_log_file(input_file, output_file='selected_methods.json'):
    """
    解析日志文件，提取block和step信息，并保存为JSON格式
    
    Args:
        input_file: 输入日志文件路径
        output_file: 输出JSON文件路径
    """
    # 初始化数据结构
    data = {}
    
    # 正则表达式模式，用于提取block和step信息
    pattern = r'block(\d+),step(\d+)\s+loss([\d\.]+)\s+threshold([\d\.]+)'
    
    # 读取文件并解析每一行
    with open(input_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                block_num = int(match.group(1))
                step_num = int(match.group(2))
                loss = float(match.group(3))
                threshold = float(match.group(4))
                
                # 确保step_num在data中有一个条目
                if str(step_num) not in data:
                    data[str(step_num)] = []
                
                # 保存block和method信息，只保存step和block对
                data[str(step_num)].append(str(block_num))
                
                print(f"已解析: block {block_num}, step {step_num}, loss {loss}, threshold {threshold}")
    
    # 保存到JSON文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"数据已保存到 {output_file}")
    return data

if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入文件路径
    input_file = os.path.join(current_dir, "1.txt")
    
    # 输出文件路径
    output_file = os.path.join(current_dir, "selected_methods.json")
    
    # 解析日志文件
    data = parse_log_file(input_file, output_file)