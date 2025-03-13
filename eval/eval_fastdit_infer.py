import os
from pathlib import Path
import soundfile as sf
import tomli
import torch
import time

from f5_tts.infer.infer_cli import infer_process, load_model, load_vocoder, DiT
from cached_path import cached_path

def load_config(config_path):
    """加载toml配置文件"""
    try:
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None

def ensure_dir(dir_path):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(dir_path, exist_ok=True)

def process_sentence_pair(src_info, tgt_info, model, vocoder, output_dir, speed, delta=None):
    """处理一对句子，生成音频并保存
    
    Returns:
        tuple: (推理时间(秒), 音频时长(秒))
    """
    ensure_dir(output_dir)
    output_path = f"{output_dir}/{tgt_info['id']}.flac"
    
    # 只对infer_process计时
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    # 生成目标句子音频
    audio,final_sample_rate, _ = infer_process(
        ref_audio=src_info['audio_path'],
        ref_text=src_info['text'],
        gen_text=tgt_info['text'],
        model_obj=model,
        vocoder=vocoder,
        mel_spec_type='vocos',
        speed=speed,
        delta = delta
    )
    
    end_event.record()
    torch.cuda.synchronize()
    infer_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
    
    with open(output_path,"wb") as f:
        sf.write(f.name, audio, final_sample_rate)
    
    # 计算音频时长(秒)
    audio_duration = len(audio) / final_sample_rate
    
    return infer_time, audio_duration

def main():
    config_path = 'eval/config/eval_config.toml'
    
    # 加载配置文件
    config = load_config(config_path)
    if config is None:
        print("使用默认配置和命令行参数")
        config = {
            "paths": {
                "lst_file": "data/LibriSpeech/librispeech_pc_test_clean_cross_sentence.lst",
                "output_dir": "data/LibriSpeech/test-clean_output",
                "ckpt_file": "",
                "vocab_file": ""
            },
            "model": {
                "name": "F5-TTS",
                "vocoder_name": "vocos",
            },
            "generation": {
                "speed": 1.0,
                "delta": 0.2
            }
        }
    
    # 命令行参数覆盖配置文件
    lst_file = config["paths"]["lst_file"]
    output_dir = config["paths"]["output_dir"]
    model_name = config["model"]["name"]
    ckpt_file = config["paths"]["ckpt_file"]
    vocab_file = config["paths"]["vocab_file"]
    vocoder_name = config["model"]["vocoder_name"]
    speed = config["generation"]["speed"]
    delta = config["generation"]["delta"]

    # 加载vocoder
    vocoder = load_vocoder(vocoder_name=vocoder_name)

    # 加载模型
    if model_name == "F5-TTS":
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        if ckpt_file == "":
            if vocoder_name == "vocos":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base"
                ckpt_step = 1200000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
            elif vocoder_name == "bigvgan":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base_bigvgan"
                ckpt_step = 1250000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))
    else:
        print("只测试F5-TTS模型.")
        return
    
    model = load_model(model_cls, model_cfg, ckpt_file, 
                      mel_spec_type=vocoder_name, 
                      vocab_file=vocab_file)
    
    # 读取lst文件
    with open(lst_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = lines[::25]
    # 处理每一对句子
    processed_pairs = 0
    total_audio_duration = 0.0
    total_infer_time = 0.0
    
    for i, line in enumerate(lines):
        try:
            # 解析行
            src_id, src_dur, src_text, tgt_id, tgt_dur, tgt_text = line.strip().split('\t')
            
            # 构建音频路径
            src_audio_path = f"data/LibriSpeech/test-clean/{src_id.split('-')[0]}/{src_id.split('-')[1]}/{src_id}.flac"
            tgt_audio_path = output_dir
            
            # 准备句子信息
            src_info = {
                'id': src_id,
                'duration': float(src_dur),
                'text': src_text,
                'audio_path': src_audio_path
            }
            
            tgt_info = {
                'id': tgt_id,
                'duration': float(tgt_dur),
                'text': tgt_text,
                'audio_path': tgt_audio_path
            }
            
            # 处理这对句子
            infer_time, audio_duration = process_sentence_pair(
                src_info, tgt_info, model, vocoder, tgt_audio_path, speed,delta
            )
            
            # 累加统计
            total_infer_time += infer_time
            total_audio_duration += audio_duration
            processed_pairs += 1
            
            print(f"Processed pair {i}/{len(lines)}: {src_id} -> {tgt_id} (infer: {infer_time:.2f}s, audio: {audio_duration:.2f}s)")
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing line {i}:\n{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            continue
    
    # 打印统计信息
    print(f'\n共处理完成{processed_pairs}对句子:')
    print(f'总推理时间: {total_infer_time:.2f} 秒')
    print(f'生成音频总时长: {total_audio_duration:.2f} 秒')
    print(f'实时率 RTF: {total_infer_time/total_audio_duration:.4f}')

if __name__ == "__main__":
    main()
