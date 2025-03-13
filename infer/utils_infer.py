# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys
import logging

from f5_tts.model.logger import Logger

from f5_tts.model.utils import LatencyBenchmark

os.environ["PYTOCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"../../{os.path.dirname(os.path.abspath(__file__))}/third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib
from f5_tts.model.hook import (
    calibration,
    set_need_cahce_residual,
    speedup,
    calculate_flops_hook,
    calculate_ff_flops_hook,
    insert_wars_to_attention_forward,
    pre_calibration,
    pre_calibration_check
)
matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from skimage.filters import threshold_otsu
from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
    threshold_gmm
)

_ref_audio_cache = {}

device = "cuda" # if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16 if "cuda" in device and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16 if "cuda" in device and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, clip_short=True, show_info=print, device=device):
    show_info("Converting audio...")
    print(ref_audio_orig)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 15000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 15000:
                aseg = aseg[:15000]
                show_info("Audio is over 15s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    if not ref_text.strip():
        global _ref_audio_cache
        if audio_hash in _ref_audio_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("ref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    calibration_mode=False,
    delta=None,
    timer = False
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text)

    # show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return infer_batch_process(
        (audio, sr),
        ref_text,
        gen_text_batches,
        model_obj,
        vocoder,
        mel_spec_type=mel_spec_type,
        progress=progress,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
        calibration_mode=calibration_mode,
        delta=delta,
        timer = timer
    )


# infer batches


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    calibration_mode=False,  
    delta=None,  
    timer=False
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    for i, gen_text in enumerate(progress.tqdm(gen_text_batches)):
        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            calibrate_hook = None
            if calibration_mode:
                # 先预校准
                pre_hooks = pre_calibration_check(model_obj)
                # 启动一次
                _ , _ = model_obj.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    seed = 42,
                    sway_sampling_coef=sway_sampling_coef,
                    delta=delta
                )
                
                # 移除hooks
                for hook in pre_hooks:
                    hook.remove()
                # 收集各个时间步、块下的相似度
                similarities = []
                similarities_asc = []
                for blocki in range(len(model_obj.transformer.transformer_blocks)):
                    attn= model_obj.transformer.transformer_blocks[blocki].attn
                    similarities_list_i = list(attn.diagonal_similarities.values())
                    similarities_asc_list_i = list(attn.diagonal_similarities_asc.values())
                    similarities.extend(similarities_list_i)
                    similarities_asc.extend(similarities_asc_list_i)
                
                # 使用 Otsu 方法计算阈值
                similarities = torch.tensor(similarities, device='cpu').numpy()
                similarities_asc = torch.tensor(similarities_asc, device='cpu').numpy()
                
                # # 指定阈值
                # threshold = 0.25
                # threshold_asc = 0.8
                
                threshold = threshold_otsu(similarities)
                threshold_asc = threshold_otsu(similarities_asc)
                print(f"Pre Calibration Otsu Threshold: {threshold}")
                print(f"Pre Calibration Otsu Threshold_asc: {threshold_asc}")
                
                ast_first_cnt = 0
                asc_first_cnt = 0
                
                for blocki in range(len(model_obj.transformer.transformer_blocks)):
                    attn = model_obj.transformer.transformer_blocks[blocki].attn
                    attn.ast_first = {}
                    attn.asc_first = {}
                    # 遍历该block的所有时间步相似度
                    for step, similarity, similarity_asc in zip(attn.diagonal_similarities.keys(), attn.diagonal_similarities.values(), attn.diagonal_similarities_asc.values()):
                        
                        ratio_b = 1 # - 0.2*(max(blocki / (len(model_obj.transformer.transformer_blocks)-0.5),0)) # 深层块应该放松
                        ratio_t = 1 # - 0.2*(max(step / (len(attn.diagonal_similarities.keys())-0.5),0)) # 后期步应该放松
                        attn.ast_first[step] = False
                        attn.asc_first[step] = False
                        if similarity >= threshold*ratio_b*ratio_t:
                            attn.ast_first[step] = True
                            ast_first_cnt += 1
                        if similarity_asc >= threshold_asc*ratio_b*ratio_t:
                            attn.asc_first[step] = False #True
                            asc_first_cnt += 1
                    # 清理相似度字典以释放内存
                    del attn.diagonal_similarities
                    del attn.diagonal_similarities_asc
                
                print(f"AST First Count: {ast_first_cnt}")
                print(f"ASC First Count: {asc_first_cnt}")
                
                # 开始预校准

                pre_calibrate_hook = pre_calibration(model_obj, steps=nfe_step, threshold=delta)

                # 预校准
                for mode in range(2,-1,-1):
                    method_dict = {
                        'full_attention':0,
                        'AST':0,
                        'ASC':0,
                        'wars+ASC':0,
                        'wars':0
                    }
                    insert_wars_to_attention_forward(model_obj.transformer,steps = nfe_step, window_ratio=0.125, is_method_init=False)
                    model_obj.transformer.calibration_mode = mode
                    _, _ = model_obj.sample(
                        cond=audio,
                        text=final_text_list,
                        duration=duration,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        delta=delta
                    )
                    for blocki, block in enumerate(model_obj.transformer.transformer_blocks):
                        for method_i in block.attn.steps_method:
                            method_dict[method_i] += 1
                    print(method_dict)

                pre_calibrate_hook.remove()
                
                # 如果是校准模式，调用calibrate方法
                calibrate_hook = calibration(model_obj, steps=nfe_step, threshold=delta, window_ratio=0.125)
            elif calibration_mode is False and delta is not None:
                speedup(model_obj, steps = nfe_step, window_ratio=0.125, delta=delta)#w
            else:
                insert_wars_to_attention_forward(model_obj.transformer,steps = nfe_step, window_ratio=0.125)#w
                
            # 统计计算的钩子
            hooks = []
            if calibrate_hook is not None:
                hooks.append(calibrate_hook)
            # 设置一些参数量
            for block in model_obj.transformer.transformer_blocks:
                block.attn.full_ops = 0
                block.attn.efficient_ops = 0
                block.ff.full_ops = 0
                block.ff.efficient_ops = 0
                # block.attention.need_cache_residual = [True] * len(block.attention.need_cache_residual)
                hook = block.attn.register_forward_pre_hook(calculate_flops_hook, with_kwargs=True)
                hook_ff = block.ff.register_forward_pre_hook(calculate_ff_flops_hook, with_kwargs=True)
                hooks.append(hook)
                hooks.append(hook_ff)

            # 正式校准
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                delta=delta
            )

            total_full_ops,total_efficient_ops = 0,0
            total_full_ops_ff,total_efficient_ops_ff = 0,0
            
            for blocki, block in enumerate(model_obj.transformer.transformer_blocks):
                print('-'*55)
                print(f'块{blocki}原需attn ops: {round(block.attn.full_ops/1e9, 4)}G，加速后attn ops: {round(block.attn.efficient_ops/1e9, 4)}G')
                total_full_ops += block.attn.full_ops
                total_efficient_ops += block.attn.efficient_ops
                
                print(f'\n块{blocki}原需ff ops: {round(block.ff.full_ops/1e9, 4)}G，加速后ff ops: {round(block.ff.efficient_ops/1e9, 4)}G')
                total_full_ops_ff += block.ff.full_ops
                total_efficient_ops_ff += block.ff.efficient_ops
            
            print('-'*55)
            print(f'总原需attn ops: {round(total_full_ops/1e9, 4)}G，加速后attn ops: {round(total_efficient_ops/1e9, 4)}G')
            print(f'总原需ff ops: {round(total_full_ops_ff/1e9, 4)}G，加速后ff ops: {round(total_efficient_ops_ff/1e9, 4)}G')
            
            for hook in hooks:
                hook.remove()

            if calibration_mode:
                transformer = model_obj.transformer
                set_need_cahce_residual(transformer)

                # 保存校准后得到的方法
                to_save_methods = {'methods': [], 'need_residual': []}
                for blocki, block in enumerate(transformer.transformer_blocks):
                    to_save_methods['methods'].append(block.attn.steps_method)
                    to_save_methods['need_residual'].append(block.attn.need_cache_residual)

                with open(f"data\\methods\\{nfe_step}_{delta}_0.125.json", 'w') as file: #w
                    import json
                    file.write(json.dumps(to_save_methods))

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated_mel_spec)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated_mel_spec)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )

            final_wave = new_wave

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    return final_wave, target_sample_rate, combined_spectrogram


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
