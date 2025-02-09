import glob
import json
import os
import traceback
import librosa
import numpy as np
import setproctitle
import torch
import yaml
import soundfile as sf
import torch.nn.functional as F

from copy import deepcopy
from multiprocessing import Queue, Process
from nemo_text_processing.text_normalization.normalize import Normalizer
from langdetect import detect as classify_language
from pydub import AudioSegment
from tqdm import tqdm

from modules.commons.nar_tts_modules import LengthRegulator
from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
from utils.text.text_encoder import TokenTextEncoder
from utils.text.ph_tone_convert import map_phone_to_tokendict, split_ph_timestamp, split_ph
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import types
from typing import Any, Optional, Tuple

from modules.tts.megatts3.flow_matching.llama import apply_rotary_emb
from flash_attn import flash_attn_func
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""
计算Attention的Flops
"""

def calculate_flops_hook(model, args, kwargs):
    hidden_states = args[0]
    batch_size, seq_len, dim = hidden_states.size()

    ops = seq_len * seq_len * model.heads * batch_size * dim // model.heads + seq_len * dim * batch_size * seq_len
    
    model.full_ops += ops

    method = model.steps_method[model.step]
    window_size = model.window_size * 2

    if method == "full_attention":
        if model.need_cache_residual[model.step]:
            ops *= 1 + window_size / seq_len
    elif method == "ASC":
        ops = ops / 2
        if model.need_cache_residual[model.step]:
            ops *= 1 + window_size / seq_len
    elif method == 'wars':
        ops *= window_size / seq_len
    elif method == 'wars+ASC':
        ops = ops * window_size / seq_len / 2
    elif method == "AST":
        ops = 0

    model.efficient_ops += ops
    
"""
计算raw output与efficient output之间的差距，使用默认值计算Loss
"""
def compression_loss(a, b, metric=""):
    ls = []
    for ai, bi in zip(a, b):
        if isinstance(ai, torch.Tensor):
            diff = (ai - bi) / (torch.max(ai, bi) + 1e-6)
            l = diff.abs().clip(0, 10).mean()
            ls.append(l)
    l = sum(ls) / len(ls)
    return l

"""
校准函数，使用贪心算出各个机制所对应的超参数
"""
def transformer_forward_pre_hook_for_calibration(model, args, kwargs):
    
    now_stepi = model.transformer_blocks[0].attn.step
    print(f"Calibration Step: {now_stepi}")

    # 为了避免在搜索candidate method时对cache内容产生改变，因此搜索时需要先关掉cache的开关
    for block in model.transformer_blocks:
        block.attn.forward = types.MethodType(efficient_attention_forward, block.attn)
        block.attn.need_cache_output[now_stepi] = False
        block.attn.need_cache_residual[now_stepi] = False

    # 先走一遍得到full-attention的值
    raw_outputs = model.forward(*args, **kwargs)
    for blocki, block in enumerate(model.transformer_blocks):
        if now_stepi == 0:
            continue
        # method的由强到弱
        method_candidates = ['AST', 'wars+ASC', 'wars', 'ASC']
        selected_method = 'full_attention'
        for method in method_candidates:
            # print(f"Try###Block:{blocki} Step:{now_stepi} Method:{method}")
            block.attn.steps_method[now_stepi] = method

            for block_ in model.transformer_blocks:
                block_.attn.step = now_stepi
            efficient_outputs = model.forward(*args, **kwargs)
            loss = compression_loss(raw_outputs, efficient_outputs)
            threshold = model.loss_thresholds[now_stepi][blocki]
            # print(f"Try### Block:{blocki} Step:{now_stepi} Method:{method} Loss:{loss} Threshold:{threshold}")

            if loss<threshold:
                selected_method = method
                break
        
        block.attn.steps_method[now_stepi] = selected_method
        del loss, efficient_outputs
    del raw_outputs

    # 因为这只是一个transformer的一个prehook，
    # 在最终确定好所有的机制以后还会走一次transformer的forward，在那一个forward里面step会递增，因此这里需要将递增的step恢复
    for block_ in model.transformer_blocks:
        block_.attn.step = now_stepi
    
    # 在确定本次Step的计划确定之后，将Cache的开关打开，使得本次Step的Cache能够正常产生
    for block in model.transformer_blocks:
        block.attn.need_cache_output[now_stepi] = True
        block.attn.need_cache_residual[now_stepi] = True

def set_need_cahce_residual(transformer):
    for blocki, block in enumerate(transformer.transformer_blocks):
        for stepi in range(len(block.attn.need_cache_residual)-1):
            if block.attn.steps_method[stepi+1] == 'full_attention':
                block.attn.need_cache_residual[stepi] = False
            elif block.attn.steps_method[stepi+1] == 'ASC':
                block.attn.need_cache_residual[stepi] = False
        block.attn.need_cache_residual[-1] = False

def calibration(wav_path, txt_fn, out_path, worker_id, device, model, steps=32, threshold=0.1, window_size=64, saved_methods_path=""):

    print("Calibration for transformer!!!")
    transformer = model.transformer # model应该是cfm

    loss_thresholds = []
    for step_i in range(steps):
        sub_list = []
        for blocki in range(len(transformer.transformer_blocks)):
            threshold_i = (blocki + 1) / len(transformer.transformer_blocks) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)

    insert_wars_to_attention_forward(transformer)
    if os.path.exists(saved_methods_path):
        import json
        saved_methods = json.loads(open(saved_methods_path).read())['methods']
        saved_need_cache_residual = json.loads(open(saved_methods_path).read())['need_residual']

        for methods, need_cache_residual, block in zip(saved_methods, saved_need_cache_residual, transformer.transformer_blocks):
            block.attn.steps_method = methods
            block.attn.need_cache_residual = need_cache_residual
            assert len(methods) == steps
            assert len(need_cache_residual) == steps
        set_need_cahce_residual(transformer)

        return

    hook = transformer.register_forward_pre_hook(transformer_forward_pre_hook_for_calibration, with_kwargs=True)
    transformer.loss_thresholds = loss_thresholds

    # convert_to_wav(wav_path)
    # wav_path = wav_path[:-4] + '.wav'
    # os.makedirs(out_path, exist_ok=True)

    # print(f"| Start Calibration {wav_path}+{txt_fn}")
    # # subprocess.check_call(f'cp "{wav_path}" "{out_path}/ref.wav"', shell=True)
    
    # # 先只采样一条做校准
    # inp_txts = [x.strip() for x in open(txt_fn).readlines()]
    # inp_txts = [x for x in inp_txts if x != '']
    # inp_txts = [inp_txts[0]]
    
    # for i, (wav_pred, sr, txt) in enumerate(infer_ins.forward_model([wav_path], inp_txts, out_path)):
    #     save_wav(wav_pred, f'{out_path}/Calibration_[P]{inp_txts[i][:20]}.wav', sr=sr)
    
    hook.remove()
    del hook
    set_need_cahce_residual(transformer)

    # 保存校准后得到的方法
    to_save_methods = {'methods': [], 'need_residual': []}
    for blocki, block in enumerate(transformer.transformer_blocks):
        to_save_methods['methods'].append(block.attn.steps_method)
        to_save_methods['need_residual'].append(block.attn.need_cache_residual)

    with open(f"saved_methods/{steps}_{threshold}_{window_size}.json", 'w') as file:
        import json
        file.write(json.dumps(to_save_methods))

def insert_wars_to_attention_forward(transformer, steps=32, window_size=64):
    methods = ["full_attention"] * len(transformer.transformer_blocks)
    output_shares = [False] * len(transformer.transformer_blocks)
    assert len(methods) == len(transformer.transformer_blocks)
    for block, method, output_share in zip(transformer.transformer_blocks, methods, output_shares):
        attn = block.attn
        # set some attribute
        attn.window_size = window_size
        attn.method = method
        attn.output_share = output_share
        attn.step = 0
        attn.forward = types.MethodType(efficient_attention_forward, attn)
        attn.steps_method = ['full_attention'] * steps
        attn.need_cache_residual = [True] * steps
        attn.need_cache_output = [True] * steps
        attn.cached_residual = None
        attn.cached_output = None

def full_forward(
    self,
    x: float["b n d"],  # noised input x  # noqa: F722
    c: float["b n d"] = None,  # context c  # noqa: F722
    mask: bool["b n"] | None = None,  # noqa: F722
    rope=None,  # rotary position embedding for x
    c_rope=None,  # rotary position embedding for c
    timestep = None,
    is_null_cond=False
) :
    if c is not None:
        return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope,block_id=self.block_id,timestep=timestep,is_null_cond=is_null_cond)
    else:
        return self.processor(self, x, mask=mask, rope=rope,block_id = self.block_id,timestep=timestep,is_null_cond=is_null_cond)

def efficient_attention_forward(            
        self,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b n d"] = None,  # context c  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
        timestep = None,
        is_null_cond=False
    ):
    method = self.steps_method[self.step]
    # print(method, self.step)

    # 是否直接share最近一个Step的output, AST机制
    if 'AST' in method:
        self.step += 1
        return self.cached_output

    # ASC机制计算
    # 如果使用了ASC机制，那我们先只算conditional的情况    
    return self.processor(self, x, mask=mask, rope=rope,block_id = self.block_id,timestep=timestep,is_null_cond=is_null_cond,method = method,need_cache_residual = self.need_cache_residual[self.step],need_cache_output = self.need_cache_output[self.step])
    
    if self.need_cache_output[self.step]:
        self.cached_output = output
    
    self.step += 1
    return output

# def efficient_reference(wav_path, txt_fn, out_path, worker_id, device, infer_ins):
#     setproctitle.setproctitle('megatts_inference_worker')
#     convert_to_wav(wav_path)
#     wav_path = wav_path[:-4] + '.wav'
#     os.makedirs(out_path, exist_ok=True)
#     try:
#         print(f"| Start processing {wav_path}+{txt_fn}")
#         inp_txts = [x.strip() for x in open(txt_fn).readlines()]
#         inp_txts = [x for x in inp_txts if x != '']
#         hooks = []
#         # 设置一些参数量
#         for block in infer_ins.diff_model.encoder.layers:
#             block.attn.step = 0
#             block.attn.full_ops = 0
#             block.attn.efficient_ops = 0
#             # block.attn.need_cache_residual = [True] * len(block.attn.need_cache_residual)
#             hook = block.attn.register_forward_pre_hook(calculate_flops_hook, with_kwargs=True)
#             hooks.append(hook)

#         total_full_ops, total_efficient_ops = 0, 0
#         for i, (wav_pred, sr, txt) in enumerate(infer_ins.forward_model([wav_path], inp_txts, out_path)):

#             save_wav(wav_pred, f'{out_path}/[Efficient]{inp_txts[i][:20]}.wav', sr=sr)

#             # 计算attn ops的变化量以及重置一些参数
#             full_ops, efficient_ops = 0, 0
#             for block in infer_ins.diff_model.encoder.layers:
#                 block.attn.step = 0
#                 full_ops += block.attn.full_ops 
#                 efficient_ops += block.attn.efficient_ops

#                 total_full_ops += block.attn.full_ops 
#                 total_efficient_ops += block.attn.efficient_ops

#                 block.attn.full_ops = 0
#                 block.attn.efficient_ops = 0
#             print(f"Attn Ops Relative to Full: {round(efficient_ops/full_ops, 4) * 100}")

#         # 将一些统计信息写入  

#         with open(f"result_{device}.json", 'a+') as write_file:
#             from collections import defaultdict
#             methodsdict = defaultdict(int)
#             for block in infer_ins.diff_model.encoder.layers:
#                 for method in block.attn.steps_method:
#                     methodsdict[method] += 1
            
#             method_ratio = []
#             for method in ['AST', 'wars+ASC', 'wars', 'ASC', 'full_attention']:
#                 method_ratio.append(methodsdict[method] / (len(block.attn.steps_method) * len(infer_ins.diff_model.encoder.layers)) )

#             write_file.write(json.dumps(
#                 {
#                     'out_path': out_path,
#                     'relative ops': round(total_efficient_ops/total_full_ops, 4) * 100,
#                     'method_ratio': method_ratio
#                 }
#             ) + "\n")
                
            
#         # 移除hooks
#         for hook in hooks:
#             hook.remove()

#     except:
#         print(f"| Error occurs when processing {wav_path}+{txt_fn}")
#         traceback.print_exc()

import threading
lock = threading.Lock()
if __name__ == '__main__':

    device_id = 0
    steps=32
    saved_methods_path = ""
    threshold_list = {0: [0.1, 0.3], 1: [0.5, 0.7]}[device_id]
    window_size_list = [8, 16, 32, 64]

    jobs = []
    prompts_path = []
    prompts_path += glob.glob('ref/*.wav')
    jobs += [
            [x, 'prompt/koubo_prompt.txt', f'prompt/gen_small_fsdp/{x.split("/")[-1][:-4]}/reference']
            for x in prompts_path
        ]
    infer_ins = MegaTTS3DiTInfer(device=f'cuda:{device_id}')

    # Full attention推理
    for ref_file, prompt_file, outputs in jobs:
        full_reference(ref_file, prompt_file, outputs, 0, 0, infer_ins)
    
    for threshold in threshold_list:
        for window_size in window_size_list:
            jobs = []
            prompts_path = []
            prompts_path += glob.glob('ref/*.wav')
            jobs += [
                    [x, 'prompt/koubo_prompt.txt', f'prompt/gen_small_fsdp/{x.split("/")[-1][:-4]}/steps{steps}_threshold{threshold}_windowsize{window_size}']
                    for x in prompts_path
                ]
            for ref_file, prompt_file, outputs in jobs:
                if os.path.exists(outputs) and len(os.listdir(outputs)) == 12:
                    print(f"Skip {outputs}")
                    continue

                calibration(ref_file, prompt_file, outputs, 0, device_id, infer_ins, saved_methods_path=saved_methods_path, steps=steps, threshold=threshold, window_size=window_size)
                efficient_reference(ref_file, prompt_file, outputs, 0, device_id, infer_ins)

