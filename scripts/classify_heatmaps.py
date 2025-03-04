import os
import shutil
import re

def classify_heatmaps(source_dir, output_dir):
    """
    对attention heatmap图片进行分类整理
    
    Args:
        source_dir: 源目录，包含cond和uncond子目录
        output_dir: 输出目录，将创建按step和block分类的文件夹
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建按step分类的目录
    step_dir = os.path.join(output_dir, "by_step")
    os.makedirs(step_dir, exist_ok=True)
    
    # 创建按block分类的目录
    block_dir = os.path.join(output_dir, "by_block")
    os.makedirs(block_dir, exist_ok=True)
    
    # 遍历cond和uncond文件夹
    for cond_type in ["cond", "uncond"]:
        cond_path = os.path.join(source_dir, cond_type)
        if not os.path.exists(cond_path):
            print(f"警告: {cond_path} 不存在")
            continue
        
        # 图片文件名模式: block_X_step_Y_cond.png 或 block_X_step_Y_uncond.png
        pattern = re.compile(r'block_(\d+)_step_(\d+)_(cond|uncond).png')
        
        # 遍历该文件夹下的所有图片
        for img_file in os.listdir(cond_path):
            match = pattern.match(img_file)
            if match:
                block_num = match.group(1)
                step_num = match.group(2)
                img_type = match.group(3)
                img_path = os.path.join(cond_path, img_file)
                
                # 按step分类
                step_subdir = os.path.join(step_dir, f"step_{step_num}")
                os.makedirs(step_subdir, exist_ok=True)
                step_type_dir = os.path.join(step_subdir, img_type)
                os.makedirs(step_type_dir, exist_ok=True)
                dest_path = os.path.join(step_type_dir, img_file)
                shutil.copy2(img_path, dest_path)
                print(f"复制 {img_path} 到 {dest_path}")
                
                # 按block分类
                block_subdir = os.path.join(block_dir, f"block_{block_num}")
                os.makedirs(block_subdir, exist_ok=True)
                block_type_dir = os.path.join(block_subdir, img_type)
                os.makedirs(block_type_dir, exist_ok=True)
                dest_path = os.path.join(block_type_dir, img_file)
                shutil.copy2(img_path, dest_path)
                print(f"复制 {img_path} 到 {dest_path}")
    
    print("\n分类完成！")
    print(f"按step分类的图片位于: {step_dir}")
    print(f"按block分类的图片位于: {block_dir}")

if __name__ == "__main__":
    # 源目录和输出目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(base_dir, "assets", "attention_heatmaps")
    output_dir = os.path.join(base_dir, "assets", "classified_heatmaps")
    
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    
    # 执行分类
    classify_heatmaps(source_dir, output_dir)
