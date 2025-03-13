import math
import os
import random
import string
import logging

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import convert_char_to_pinyin


# seedtts testset metainfo: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_seedtts_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


# librispeech test-clean metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        # ref_txt = ref_txt[0] + ref_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        # gen_txt = gen_txt[0] + gen_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
        gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo


# padded to max length mel batch
def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = padded_ref_mels.permute(0, 2, 1)
    return padded_ref_mels


# get prompts from metainfo containing: utt, prompt_text, prompt_wav, gt_text, gt_wav


def get_inference_prompt(
    metainfo,
    speed=1.0,
    tokenizer="pinyin",
    polyphone=True,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="vocos",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=3,
    max_secs=40,
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    for utt, prompt_text, prompt_wav, gt_text, gt_wav in tqdm(metainfo, desc="Processing prompts..."):
        # Audio
        ref_audio, ref_sr = torchaudio.load(prompt_wav)
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
        if ref_rms < target_rms:
            ref_audio = ref_audio * target_rms / ref_rms
        assert ref_audio.shape[-1] > 5000, f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio)

        # Text
        if len(prompt_text[-1].encode("utf-8")) == 1:
            prompt_text = prompt_text + " "
        text = [prompt_text + gt_text]
        if tokenizer == "pinyin":
            text_list = convert_char_to_pinyin(text, polyphone=polyphone)
        else:
            text_list = text

        # Duration, mel frame length
        ref_mel_len = ref_audio.shape[-1] // hop_length
        if use_truth_duration:
            gt_audio, gt_sr = torchaudio.load(gt_wav)
            if gt_sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(gt_sr, target_sample_rate)
                gt_audio = resampler(gt_audio)
            total_mel_len = ref_mel_len + int(gt_audio.shape[-1] / hop_length / speed)

            # # test vocoder resynthesis
            # ref_audio = gt_audio
        else:
            ref_text_len = len(prompt_text.encode("utf-8"))
            gen_text_len = len(gt_text.encode("utf-8"))
            total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len / speed)

        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)

        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        assert (
            min_tokens <= total_mel_len <= max_tokens
        ), f"Audio {utt} has duration {total_mel_len*hop_length//target_sample_rate}s out of range [{min_secs}, {max_secs}]."
        bucket_i = math.floor((total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets)

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        total_mel_lens[bucket_i].append(total_mel_len)
        final_text_list[bucket_i].extend(text_list)

        batch_accum[bucket_i] += total_mel_len

        if batch_accum[bucket_i] >= infer_batch_size:
            # print(f"\n{len(ref_mels[bucket_i][0][0])}\n{ref_mel_lens[bucket_i]}\n{total_mel_lens[bucket_i]}")
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                total_mel_lens[bucket_i],
                final_text_list[bucket_i],
            ) = [], [], [], [], [], []

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all


# get wav_res_ref_text of seed-tts test metalst
# https://github.com/BytedanceSpeech/seed-tts-eval


def get_seed_tts_test(metalst, gen_wav_dir, gpus):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")

        if not os.path.exists(os.path.join(gen_wav_dir, utt + ".wav")):
            continue
        gen_wav = os.path.join(gen_wav_dir, utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        test_set_.append((gen_wav, prompt_wav, gt_text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# get librispeech test-clean cross sentence test


def get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    # 取20个
    lines = lines[::10]

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        if eval_ground_truth:
            gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
        else:
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".flac")):
                raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".flac")

        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")
        # gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
        # ref_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# load asr model


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model


# WER Evaluation, the way Seed-TTS does


def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)

    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation
    wers = []

    from jiwer import compute_measures

    index = 1
    for gen_wav, prompt_wav, truth in test_set:
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
            hypo = ""
            for segment in segments:
                hypo = hypo + " " + segment.text

        # raw_truth = truth
        # raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()

        measures = compute_measures(truth, hypo)
        wer = measures["wer"]

        # ref_list = truth.split(" ")
        # subs = measures["substitutions"] / len(ref_list)
        # dele = measures["deletions"] / len(ref_list)
        # inse = measures["insertions"] / len(ref_list)

        wers.append(wer)
        print(f"{index}/{len(test_set)}: {wer}")

        index += 1
    
    print(f'本批WER: {round(np.mean(wers) * 100, 3)}')

    return wers


import numpy as np
import librosa

def calculate_lsd(audio1_path, audio2_path, sr=16000, n_fft=512):
    """
    计算两段音频之间的 Log-Spectral Distance (LSD)
    
    参数:
        audio1_path: 参考音频文件路径
        audio2_path: 生成音频文件路径
        sr: 采样率（默认16000 Hz）
        n_fft: FFT窗口大小（默认512）
    
    返回:
        lsd: Log-Spectral Distance 值
    """
    # 加载音频文件
    audio1, _ = librosa.load(audio1_path, sr=sr)
    audio2, _ = librosa.load(audio2_path, sr=sr)
    
    # 确保两段音频长度一致（截取到较短的长度）
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # 计算功率谱密度 (Power Spectral Density, PSD)
    S1 = np.abs(librosa.stft(audio1, n_fft=n_fft))**2
    S2 = np.abs(librosa.stft(audio2, n_fft=n_fft))**2
    
    # 转换为对数尺度
    log_S1 = 10 * np.log10(S1 + 1e-10)  # 防止除以零
    log_S2 = 10 * np.log10(S2 + 1e-10)
    
    # 计算 LSD
    lsd = np.sqrt(np.mean((log_S1 - log_S2)**2))
    print(f"LSD: {lsd:.4f}")
    return lsd

# SIM Evaluation


def run_sim(args):
    rank, test_set, ckpt_dir = args
    logger = logging.getLogger("eval")
    device = f"cuda:{rank}"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sim_list = []
    i = 0
    for wav1, wav2, truth in test_set:
        wav1, sr1 = torchaudio.load(wav1)
        wav2, sr2 = torchaudio.load(wav2)

        resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
        resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
        logger.info(f"{i}/{len(test_set)}: {sim:.4f}")
        sim_list.append(sim)
        i += 1

    sim = round(sum(sim_list) / len(sim_list), 3)
    print(f'本批SIM: {sim}')
    return sim_list
