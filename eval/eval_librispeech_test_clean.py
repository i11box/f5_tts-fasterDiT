# Evaluate with Librispeech test-clean, ~3s prompt to generate 4-10s audio (the way of valle/voicebox evaluation)

import sys
import os

sys.path.append(os.getcwd())

import numpy as np

from f5_tts.eval.utils_eval import (
    get_librispeech_test,
    run_asr_wer,
    run_sim,
)

eval_task = "sim"  # sim | wer
lang = "en"
metalst = "data\\Librispeech\\librispeech_pc_test_clean_cross_sentence.lst"
librispeech_test_clean_path = "data\\LibriSpeech\\test-clean"  # test-clean path
gen_wav_dir = "data\\LibriSpeech\\test-clean_output"  # generated wavs

gpus = [0]
test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path)

## In LibriSpeech, some speakers utilized varying voice characteristics for different characters in the book,
## leading to a low similarity for the ground truth in some cases.
# test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth = True)  # eval ground truth

local = False
if local:  # use local custom checkpoint dir
    asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
else:
    asr_ckpt_dir = ""  # auto download to cache dir

wavlm_ckpt_dir = "data\\checkpoints\\wavlm_large_finetune.pth"


# --------------------------- WER ---------------------------

if eval_task == "wer":
    wers = []

    for rank, sub_test_set in test_set:
        wers_ = run_asr_wer((rank, lang, sub_test_set, asr_ckpt_dir))
        wers.extend(wers_)

    wer = round(np.mean(wers) * 100, 3)
    print(f"\nTotal {len(wers)} samples")
    print(f"WER      : {wer}%")


# --------------------------- SIM ---------------------------

if eval_task == "sim":
    sim_list = []

    for rank, sub_test_set in test_set:
        sim_ = run_sim((rank, sub_test_set, wavlm_ckpt_dir))
        sim_list.extend(sim_)

    sim = round(sum(sim_list) / len(sim_list), 3)
    print(f"\nTotal {len(sim_list)} samples")
    print(f"SIM      : {sim}")
