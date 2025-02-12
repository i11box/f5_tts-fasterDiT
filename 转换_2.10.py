import json

def convert_strategy_format():
    # 读取原始json
    with open('32_0.5_64.json', 'r') as f:
        source = json.load(f)
    
    # 读取method0.5.json获取时间步序列
    with open('method0.5.json', 'r') as f:
        target_format = json.load(f)
        timesteps = list(target_format['0'].keys())
    
    # 创建新格式
    result = {}
    methods_list = source['methods']
    
    # 遍历每个块
    for block_id in range(len(methods_list)):
        block_methods = methods_list[block_id]
        result[str(block_id)] = {}
        
        # 将策略映射到时间步
        for step_idx, timestep in enumerate(timesteps):
            if step_idx < len(block_methods):
                method = block_methods[step_idx].lower()
                # 转换策略名称
                if method == 'full_attention':
                    method = 'none'
                elif method == 'wars+asc':
                    method = 'asc-wars'
                result[str(block_id)][timestep] = method
            else:
                result[str(block_id)][timestep] = 'none'
    
    # 保存结果
    with open('converted_method.json', 'w') as f:
        json.dump(result, f, indent=4)

convert_strategy_format()