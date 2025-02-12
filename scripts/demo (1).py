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

    ops = seq_len * seq_len * model.encoder_n_kv_heads * batch_size * dim // model.encoder_n_kv_heads + seq_len * dim * batch_size * seq_len
    
    model.full_ops += ops

    method = model.steps_method[model.step]
    window_size = model.window_size * 2

    if method == "full_attention":
        if model.need_cache_residual[model.step]:
            ops *= 1 + window_size / seq_len
    elif method == "ASC":
        ops = ops / 3 * 2
        if model.need_cache_residual[model.step]:
            ops *= 1 + window_size / seq_len
    elif method == 'wars':
        ops *= window_size / seq_len
    elif method == 'wars+ASC':
        ops = ops * window_size / seq_len / 3 * 2
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
    
    now_stepi = model.layers[0].attention.step
    print(f"Calibration Step: {now_stepi}")

    # 为了避免在搜索candidate method时对cache内容产生改变，因此搜索时需要先关掉cache的开关
    for block in model.layers:
        block.attention.forward = types.MethodType(efficient_attention_forward, block.attention)
        block.attention.need_cache_output[now_stepi] = False
        block.attention.need_cache_residual[now_stepi] = False

    # 先走一遍得到full-attention的值
    raw_outputs = model.forward(*args, **kwargs)
    for blocki, block in enumerate(model.layers):
        if now_stepi == 0:
            continue
        # method的由强到弱
        method_candidates = ['AST', 'wars+ASC', 'wars', 'ASC']
        selected_method = 'full_attention'
        for method in method_candidates:
            # print(f"Try###Block:{blocki} Step:{now_stepi} Method:{method}")
            block.attention.steps_method[now_stepi] = method

            for block_ in model.layers:
                block_.attention.step = now_stepi
            efficient_outputs = model.forward(*args, **kwargs)
            loss = compression_loss(raw_outputs, efficient_outputs)
            threshold = model.loss_thresholds[now_stepi][blocki]
            # print(f"Try### Block:{blocki} Step:{now_stepi} Method:{method} Loss:{loss} Threshold:{threshold}")

            if loss<threshold:
                selected_method = method
                break
        
        block.attention.steps_method[now_stepi] = selected_method
        print(f"Selected### Block:{blocki} Step:{now_stepi} Method:{selected_method}")
        del loss, efficient_outputs
    del raw_outputs

    # 因为这只是一个transformer的一个prehook，
    # 在最终确定好所有的机制以后还会走一次transformer的forward，在那一个forward里面step会递增，因此这里需要将递增的step恢复
    for block_ in model.layers:
        block_.attention.step = now_stepi
    
    # 在确定本次Step的计划确定之后，将Cache的开关打开，使得本次Step的Cache能够正常产生
    for block in model.layers:
        block.attention.need_cache_output[now_stepi] = True
        block.attention.need_cache_residual[now_stepi] = True

# 每次diffusion model推理前，都需要使用hook函数重置模型的step
def diffusion_forward_pre_hook_for_reset_step(model, args, kwargs):
    print("Reset for diffusion model inference")
    for block in model.encoder.layers:
        block.attention.step = 0

def set_need_cahce_residual(transformer):
    for blocki, block in enumerate(transformer.layers):
        for stepi in range(len(block.attention.need_cache_residual)-1):
            if block.attention.steps_method[stepi+1] == 'full_attention':
                block.attention.need_cache_residual[stepi] = False
            elif block.attention.steps_method[stepi+1] == 'ASC':
                block.attention.need_cache_residual[stepi] = False
        block.attention.need_cache_residual[-1] = False

def calibration(wav_path, txt_fn, out_path, worker_id, device, infer_ins, steps=32, threshold=0.1, window_size=64, saved_methods_path=""):

    print("Calibration for transformer!!!")
    transformer = infer_ins.diff_model.encoder

    loss_thresholds = []
    for step_i in range(steps):
        sub_list = []
        for blocki in range(len(transformer.layers)):
            threshold_i = (blocki + 1) / len(transformer.layers) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)

    insert_wars_to_attention_forward(transformer)
    if os.path.exists(saved_methods_path):
        import json
        saved_methods = json.loads(open(saved_methods_path).read())['methods']
        saved_need_cache_residual = json.loads(open(saved_methods_path).read())['need_residual']

        for methods, need_cache_residual, block in zip(saved_methods, saved_need_cache_residual, transformer.layers):
            block.attention.steps_method = methods
            block.attention.need_cache_residual = need_cache_residual
            assert len(methods) == steps
            assert len(need_cache_residual) == steps
        set_need_cahce_residual(transformer)

        return


    print(transformer)
    hook = transformer.register_forward_pre_hook(transformer_forward_pre_hook_for_calibration, with_kwargs=True)
    transformer.loss_thresholds = loss_thresholds

    convert_to_wav(wav_path)
    wav_path = wav_path[:-4] + '.wav'
    os.makedirs(out_path, exist_ok=True)

    print(f"| Start Calibration {wav_path}+{txt_fn}")
    # subprocess.check_call(f'cp "{wav_path}" "{out_path}/ref.wav"', shell=True)
    
    # 先只采样一条做校准
    inp_txts = [x.strip() for x in open(txt_fn).readlines()]
    inp_txts = [x for x in inp_txts if x != '']
    inp_txts = [inp_txts[0]]
    
    for i, (wav_pred, sr, txt) in enumerate(infer_ins.forward_model([wav_path], inp_txts, out_path)):
        save_wav(wav_pred, f'{out_path}/Calibration_[P]{inp_txts[i][:20]}.wav', sr=sr)
    
    hook.remove()
    del hook
    set_need_cahce_residual(transformer)

    to_save_methods = {'methods': [], 'need_residual': []}
    for blocki, block in enumerate(transformer.layers):
        print(f"Block:{blocki}; Methods:{block.attention.steps_method}")
        to_save_methods['methods'].append(block.attention.steps_method)
        print(block.attention.need_cache_residual)
        to_save_methods['need_residual'].append(block.attention.need_cache_residual)

    with open(f"saved_methods/{wav_path.split('/')[-1]}_{inp_txts[0][:20]}_{steps}_{threshold}_{window_size}.json", 'w') as file:
        import json
        file.write(json.dumps(to_save_methods))

    # hook = infer_ins.diff_model.register_forward_pre_hook(diffusion_forward_pre_hook_for_reset_step, with_kwargs=True)


def insert_wars_to_attention_forward(model, steps=32, window_size=64):
    methods = ["full_attention"] * len(model.layers)
    output_shares = [False] * len(model.layers)
    assert len(methods) == len(model.layers)
    for block, method, output_share in zip(model.layers, methods, output_shares):
        attn = block.attention
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


def full_attention_forward(            
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        use_cache: bool = False,
    ):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    if self.use_qk_norm:
        xq = self.q_norm(xq.float()).to(xq.dtype)
        xk = self.k_norm(xk.float()).to(xk.dtype)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    keys = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    values = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    query_states = xq.transpose(1, 2)
    attention_mask = mask
    key_states = keys.transpose(1, 2)
    value_states = values.transpose(1, 2)
    output = flash_attn_func(query_states, key_states, value_states, causal=self.use_causal_attn)
    output = self.wo(output.contiguous().view(bsz, seqlen, -1))
    return output

def efficient_attention_forward(            
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        use_cache: bool = False,
    ):
    method = self.steps_method[self.step]
    # print(method, self.step)

    # 是否直接share最近一个Step的output, AST机制
    if 'AST' in method:
        self.step += 1
        return self.cached_output

    # ASC机制计算
    # 如果使用了ASC机制，那我们先只算conditional的情况    
    if 'ASC' in method:
        # 将unconditional排除
        x = x[:2, :, :]


    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    if self.use_qk_norm:
        xq = self.q_norm(xq.float()).to(xq.dtype)
        xk = self.k_norm(xk.float()).to(xk.dtype)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    keys = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    values = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    query_states = xq.transpose(1, 2)
    attention_mask = mask
    key_states = keys.transpose(1, 2)
    value_states = values.transpose(1, 2)
    """
    原本是Cache 函数，但是这里不是生成任务是表征任务，不需要Cache
    """
    # step 1，计算window_size
    w_output = flash_attn_func(query_states, key_states, value_states, causal=self.use_causal_attn, window_size=(-self.window_size, self.window_size))

    # step2，确定使用full attention还是使用wars
    if 'full_attention' in method:
        # 默认使用了full_attention就一定会计算residual
        f_output = flash_attn_func(query_states, key_states, value_states, causal=self.use_causal_attn)
        w_residual = f_output-w_output
        if self.need_cache_residual[self.step]:
            self.cached_residual = w_residual
        output = f_output
    elif 'wars' in method:
        assert hasattr(self, 'cached_residual'), "必须要先过Full attention产生Residual output才能使用Wars"
        output = w_output + self.cached_residual[:bsz]
    elif 'ASC' in method:
        f_output = flash_attn_func(query_states, key_states, value_states, causal=self.use_causal_attn)
        w_residual = f_output-w_output
        if self.need_cache_residual[self.step]:
            self.cached_residual =  torch.cat([w_residual, w_residual[1].unsqueeze(0)], dim=0)
        output = f_output
    else:
        raise NotImplementedError

    output = self.wo(output.contiguous().view(bsz, seqlen, -1))

    if 'ASC' in method:
        # 将cond_spk_txt复制给unconditional
        output = torch.cat([output, output[1].unsqueeze(0)], dim=0)
    
    if self.need_cache_output[self.step]:
        self.cached_output = output
    
    self.step += 1
    return output


    

def convert_to_wav(wav_path):
    # Check if the file exists
    if not os.path.exists(wav_path):
        print(f"The file '{wav_path}' does not exist.")
        return

    # Check if the file already has a .wav extension
    if not wav_path.endswith(".wav"):
        # Define the output path with a .wav extension
        out_path = os.path.splitext(wav_path)[0] + ".wav"

        # Load the audio file using pydub and convert it to WAV
        audio = AudioSegment.from_file(wav_path)
        audio.export(out_path, format="wav")

        print(f"Converted '{wav_path}' to '{out_path}'")

def read_wav(wav_path, sr=24000):
    try:
        y, sr_native = sf.read(wav_path)
    except sf.SoundFileRuntimeError as exc:
        raise exc

    # Final cleanup for dtype and contiguity
    y = librosa.core.to_mono(y.transpose())

    if sr is not None:
        y = librosa.core.resample(y, orig_sr=sr_native, target_sr=sr, res_type="soxr_hq")
    else:
        sr = sr_native

    return y, sr


class MegaTTS3DiTInfer():
    def __init__(
            self, 
            device=None, 
            dit_exp_name='1106_megatts3_dit_small_fsdp_stride4',
            frontend_exp_name='0830_frontendlm_asr2_stage2_init',
            vae_exp_name='1009_sdvae_lat24_stride4_left_pad_ema2_fix',
            vocoder_exp_name='1117_melgan-1117_melgan',
            dur_ckpt_path='1126_mtdurlm_fastds+balancesi_2_finetuneph_2',
            g2p_exp_name='1022_frontend_text_g2p_full_max60k_small',
            **kwargs
        ):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.dit_exp_name = dit_exp_name
        self.frontend_exp_name = frontend_exp_name
        self.vae_exp_name = vae_exp_name
        self.vocoder_exp_name = vocoder_exp_name
        self.dur_exp_name = dur_ckpt_path
        self.g2p_exp_name = g2p_exp_name
            
        # build models
        self.build_model(self.device)

        self.change_wav_header = False
        self.normalizer = Normalizer(input_case='cased', lang='en')

    def build_model(self, device):
        self.device = device
        self.precision = torch.float16

        set_hparams(exp_name=self.frontend_exp_name, print_hparams=False)

        ''' Load Dict '''
        ling_dict = json.load(open(f"./dict.json"))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
        self.token_encoder = token_encoder = self.ling_dict['phone']
        ph_dict_size = len(token_encoder)

        ''' Load AR Duration Predictor '''
        from modules.tts.megatts3.ar_dur.ar_dur_predictor import ARDurPredictor
        hp_dur_model = self.hp_dur_model = set_hparams(f'./checkpoints/{self.dur_exp_name}/config.yaml', global_hparams=False)
        # hp_dur_model['frames_multiple'] = hparams['frames_multiple']
        self.dur_model = ARDurPredictor(
            hp_dur_model, hp_dur_model['dur_txt_hs'], hp_dur_model['dur_model_hidden_size'],
            hp_dur_model['dur_model_layers'], ph_dict_size,
            hp_dur_model['dur_code_size'],
            use_rot_embed=hp_dur_model.get('use_rot_embed', False))
        self.length_regulator = LengthRegulator()
        load_ckpt(self.dur_model, f'./checkpoints/{self.dur_exp_name}', 'dur_model')
        self.dur_model.eval()
        self.dur_model.to(device)
        # self.dur_model.to(self.precision)
        self.device = device

        ''' Load Latent Diffusion '''
        self.hp_diff = hp_diff = set_hparams(f'./checkpoints/{self.dit_exp_name}/config.yaml', global_hparams=False)
        from modules.tts.megatts3.flow_matching.speechdit import Diffusion, ModelArgs
        config = ModelArgs()
        config.target_type = 'epsilon' if hp_diff.get('use_ddpm', False) else 'velocity'
        config.target_type = 'vector_field' if hp_diff.get('use_vpcfm', False) else config.target_type
        config.use_expand_ph = hp_diff.get('use_expand_ph', False)
        config.zero_xt_prompt = hp_diff.get('zero_xt_prompt', False)
        config.use_qk_norm = hp_diff.get('use_qk_norm', False)
        config.use_bpe = hp_diff.get('use_bpe', False)
        config.use_instruction = False
        config.use_adapter = False
        config.use_cache = False
        hp_vae = set_hparams(f'./checkpoints/{self.vae_exp_name}/config.yaml', global_hparams=False)
        config.in_channels = config.out_channels = hp_vae['latent_dim']
        if hp_diff.get('use_dit_large', False):
            config.encoder_dim = 1280
            config.encoder_n_layers = 36
            config.encoder_n_heads = 20
        if hp_diff.get('use_dit_xl', False):
            config.encoder_dim = 1600
            config.encoder_n_layers = 48
            config.encoder_n_heads = 25
        if hp_diff.get('use_dit_7b', False):
            config.encoder_dim = 3584
            config.encoder_n_layers = 28
            config.encoder_n_heads = 28
        self.diff_model = Diffusion(config)
        self.cfg_mask_token_phone = config.n_phone - 1
        self.cfg_mask_token_tone = config.n_tone - 1
        self.vae_stride = hp_diff.get('vae_stride', 8)
        load_ckpt(self.diff_model, f'./checkpoints/{self.dit_exp_name}/model_ckpt_steps_300000.ckpt', 'diff_model', strict=True)
        self.diff_model.eval()
        self.diff_model.to(device)
        self.diff_model.to(self.precision)

        ''' Load Llama2Tokenizer '''
        from transformers import AutoTokenizer         
        self.tokenizer = AutoTokenizer.from_pretrained("./checkpoints/llama_tokenizer", padding_side="right")
        self.tokenizer.add_tokens(['[ASR_BOS]'], special_tokens=True)
        self.tokenizer.add_tokens(['[ASR_EOS]'], special_tokens=True)
        self.tokenizer.add_tokens(['[FULL]'], special_tokens=True)
        self.tokenizer.add_tokens(['[PARTIAL]'], special_tokens=True)

        ''' Load Frontend LM '''
        from modules.tts.megatts3.frontend_lm.frontend_lm_2 import Frontend_LLAMA_Interface, LLAMA_Config
        config = LLAMA_Config()
        config.ph_vocab_size = 12800
        config.out_vocab_size = 45000
        config.ph_shift = 32200
        config.encoder_n_layers = 12
        config.phone_timestamp_start = self.phone_timestamp_start = 12798
        config.phone_timestamp_end = self.phone_timestamp_end = 12799
        config.bpe_pad = self.tokenizer.pad_token_id
        config.use_qk_norm = True
        config.use_whisper = hparams.get('use_whisper', False)
        config.flm_stage_2 = hparams.get('flm_stage_2', False)
        self.frontend_lm = Frontend_LLAMA_Interface(config)
        load_ckpt(self.frontend_lm, f'./checkpoints/{self.frontend_exp_name}', 'model')
        self.frontend_lm.eval()
        self.frontend_lm.to(device)
        self.frontend_lm.to(self.precision)

        ''' Load G2P LM'''
        from modules.tts.megatts3.frontend_lm.frontend_lm_g2p import Frontend_G2P_Interface
        config = LLAMA_Config()
        config.encoder_dim = 512
        config.encoder_n_layers = 8
        config.bpe_pad = self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # pad_token = eos_token
        config.use_qk_norm = True
        self.g2p_lm = Frontend_G2P_Interface(config)
        load_ckpt(self.g2p_lm, f'./checkpoints/{self.g2p_exp_name}', 'model')
        self.g2p_lm.eval()
        self.g2p_lm.to(device)
        self.g2p_lm.to(self.precision)
        
        ''' Wav VAE '''
        self.latent_dim = 24
        from modules.tts.megatts3.latent2wav.megatts3_wavvae import WavVAEGenerator
        hp_latent2wav = set_hparams(f'./checkpoints/1117_melgan-nsf_full_1/config.yaml', global_hparams=False)
        self.vocoder = WavVAEGenerator(self.latent_dim, hp_latent2wav, True)
        load_ckpt(self.vocoder.encoder, f'./checkpoints/{self.vae_exp_name}', 'model.module.encoder')
        load_ckpt(self.vocoder.latent2mel, f'./checkpoints/{self.vae_exp_name}', 'model.module.decoder')
        load_ckpt(self.vocoder.mel2wav, f'./checkpoints/1117_melgan-nsf_full_1', 'model_gen')
        self.vocoder.eval()
        self.vocoder.to(device)
        # self.vocoder.to(self.precision)

        ''' Whisper '''
        from funasr import AutoModel
        self.asr_model = AutoModel(
            model="checkpoints/SenseVoiceSmall",
            # trust_remote_code=True,
            # remote_code="./model.py",  
            # vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=self.device,
        )

    def audio2mel(self, audio):
        mel_basis = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
        stft = librosa.stft(audio, n_fft=400, hop_length=160, window="hann")
        magnitudes = np.abs(stft[..., :-1]) ** 2
        mel_spec = mel_basis @ magnitudes
        log_spec = np.log10(np.maximum(1e-10, mel_spec))
        log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
    
    def g2p(self, text_inp):
        with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
            text_inp = '[ASR_BOS]' + '[FULL]' + text_inp + '[ASR_EOS]'
            token_dict = self.tokenizer(text_inp, return_tensors="pt", padding=True).to(self.device)
            bpe_tokens = token_dict['input_ids']
            bpe_lengths = token_dict['attention_mask'].sum(dim=-1).long()
            phone_tokens = torch.LongTensor([798])[None, ...].to(self.device)
            phone_len = torch.LongTensor([1]).to(self.device)
            with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                tokens_pred = self.g2p_lm.g2p(bpe_tokens, bpe_lengths, phone_tokens, phone_len)
            ph_pred, tone_pred = split_ph(tokens_pred[0])
            ph_pred, tone_pred = ph_pred[None, :].to(self.device), tone_pred[None, :].to(self.device)
        return ph_pred, tone_pred

    def forward_model(self, wav_paths, inp_txts, out_path, topk_dur=1, dur_disturb=0.1, dur_alpha=1.0):
        wav_path = wav_paths[0]
        device = self.device
        sr_out = sr = 24000
        max_ph_per_sent = 100
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
        for d in devices:
            os.system(f'pkill -f "voidgpu{d}"')

        """
        1. 首先ASR识别zero-shot语音中包含的文本并将文本转换为音素
        """
        
        res = self.asr_model.generate(
            input=wav_path,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            # merge_vad=True,  #
            merge_length_s=15,
            ban_emo_unk=True,
        )
        asr_results = res[0]["text"].split('>')[-1]
        ph_g2p, tone_g2p = self.g2p(asr_results)                                                            # token转换，将输入文本经过语言模型进行编码，然后量化成另一些tokens
                                                                                                            # ph代表音素
        merged_phone = map_phone_to_tokendict({'txt_token': ph_g2p, 'tone': tone_g2p}, pad_bos_eos=False)   # 转换id token为文本

        ''' Process ref text and wav '''
        print("| Processing ", wav_path)

        """
        2. 处理wav信息
        """

        wav, _ = librosa.core.load(wav_path, sr=sr)
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:  # padding到一个固定的窗口长度
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)

        wav = whisper_wav = wav
        # wav = whisper_wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
        save_wav(wav, f'{out_path}/ref.wav', sr=sr)
        wav = torch.Tensor(wav)[None]
        mel_prompt = self.vocoder.wav2mel(wav).to(device)
        max_mel_len = mel_prompt.size(1) // 8 * 8
        mel_prompt = mel_prompt[:, :max_mel_len]
        mel_prompt_len = torch.LongTensor([x.shape[0] for x in mel_prompt]).to(device)
        
        ''' Extract speech-text alignment for mel '''
        text_inp = '[ASR_BOS]' + '[FULL]' + asr_results + '[ASR_EOS]'
        token_dict = self.tokenizer(text_inp, return_tensors="pt", padding=True).to(self.device)
        bpe_tokens = token_dict['input_ids']
        bpe_lengths = token_dict['attention_mask'].sum(dim=-1).long()
        phone_tokens = torch.LongTensor([self.phone_timestamp_start])[None, ...].to(device)
        phone_len = torch.LongTensor([1]).to(device)
        whisper_wav = librosa.resample(whisper_wav.astype(np.float32), orig_sr=24000, target_sr=16000)      # wav 进行重采样
        whisper_mel = torch.FloatTensor(self.audio2mel(whisper_wav)).to(self.device)[None].transpose(1,2)        # wav 转 mel
        max_whisper_mel_len = whisper_mel.size(1) // 8 * 8                                                  # 转换成8的倍数，这里不做pad做裁剪
        whisper_mel = whisper_mel[:, :max_whisper_mel_len]                                                  
        whisper_feat_len = torch.LongTensor([whisper_mel.size(1)]).to(device)

        ph_g2p, tone_g2p = self.g2p(asr_results)
        merged_phone = map_phone_to_tokendict({'txt_token': ph_g2p, 'tone': tone_g2p}, pad_bos_eos=False)

        with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):   # 预测音素对应的时间信息
            tokens_ref = self.frontend_lm.predict_ph_timestamp(bpe_tokens, bpe_lengths, whisper_mel, whisper_feat_len, phone_tokens, phone_len, use_sampling=False, given_phone=merged_phone)

        ''' Prepare prompt duration '''
        ph_ref, tone_ref, dur_ref, _ = split_ph_timestamp(deepcopy(tokens_ref)[0])
        ph_ref = torch.Tensor(ph_ref)[None].to(self.device)
        tone_ref = torch.Tensor(tone_ref)[None].to(self.device)
        if dur_ref.sum() < mel_prompt.size(1):
            dur_ref[-1] += mel_prompt.size(1) - dur_ref.sum()
        elif dur_ref.sum() > mel_prompt.size(1):
            len_diff = dur_ref.sum() - mel_prompt.size(1)
            while True:
                for i in range(len(dur_ref)):
                    dur_ref[i] -= 1
                    len_diff -= 1
                    if len_diff == 0:
                        break
                if len_diff == 0:
                    break
        mel2ph_ref = self.length_regulator(dur_ref[None]).to(self.device)

        if topk_dur > 1:
            self.dur_model.hparams["infer_top_k"] = topk_dur
        else:
            self.dur_model.hparams["infer_top_k"] = None

        token_encoder = self.token_encoder
        vqs = hparams['vq_stride']

        with torch.inference_mode():
            ''' Forward VAE to obtain: prompt latent '''
            ''' 使用VAE编码输入语音的mel信息 '''
            ''' 输出是 vae_latent '''
            vae_latent = self.vocoder.mel2latent(mel_prompt)
            latent_lengths = torch.LongTensor([vae_latent.size(1)]).to(device)
        
            ''' Duration Prompting '''
            dur_tokens_2d_ = mel2token_to_dur(mel2ph_ref, ph_ref.shape[1]).clamp(
                    max=self.hp_dur_model['dur_code_size'] - 1) + 1
 
            ctx_dur_tokens = dur_tokens_2d_.clone().flatten(0, 1).to(self.device)
            txt_tokens_flat_ = ph_ref.flatten(0, 1)
            ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat_ > 0][None]

            last_dur_pos_prompt = ctx_dur_tokens.shape[1]
            dur_spk_pos_ids_flat = range(0, last_dur_pos_prompt)
            dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(mel2ph_ref.device)
            with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                _, incremental_state_dur_prompt = self.dur_model.infer(
                    ph_ref, {'tone': tone_ref}, None, None, None,
                    ctx_vqcodes=ctx_dur_tokens, spk_pos_ids_flat=dur_spk_pos_ids_flat, return_state=True)
            
            ''' Generating '''
            for i, text in enumerate(tqdm(inp_txts)):
                
                if classify_language(text) == 'en':
                    text = self.normalizer.normalize(text, verbose=False, punct_post_process=False)
                
                ''' G2P '''
                ph_pred, tone_pred = self.g2p(text)

                ''' Duration Prediction '''
                ''' 预测输出文本的持续时间, 参数包含
                1. 输出文本相关信息
                2. 样本语音相关信息
                '''
                last_dur_token = ctx_dur_tokens[:, -1:]
                last_dur_pos = last_dur_pos_prompt
                incremental_state_dur = deepcopy(incremental_state_dur_prompt)
                with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                    
                    txt_len = ph_pred.shape[1]
                    dur_spk_pos_ids_flat = range(last_dur_pos, last_dur_pos + txt_len)
                    dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(device)
                    last_dur_pos = last_dur_pos + txt_len
                    dur_pred = self.dur_model.infer(
                        ph_pred, {'tone': tone_pred}, None, None, None,
                        incremental_state=incremental_state_dur,
                        first_decoder_inp=last_dur_token,
                        spk_pos_ids_flat=dur_spk_pos_ids_flat,
                    )

                    dur_pred = dur_pred - 1
                    dur_pred = dur_pred.clamp(0, self.hp_dur_model['dur_code_size'] - 1)
                    dur_pred[:, -1] = dur_pred[:, -1].clamp(64, 128)
                    # ['。', '！', '？', 'sil']
                    for sil_token in [148, 153, 166, 145]:
                        dur_pred[ph_pred==sil_token].clamp_min(32)
                    # ['，', '；'] 
                    for sil_token in [163, 165]:
                        dur_pred[ph_pred==sil_token].clamp_min(16)

                with torch.autocast(device_type="cuda", dtype=self.precision, enabled=True):
                    ''' DiT target speech generation '''
                    dur_disturb_choice = (torch.rand_like(dur_pred.float()) > 0.5).float()
                    dur_disturb_r = 1 + torch.rand_like(dur_pred.float()) * dur_disturb
                    dur_pred = dur_pred * dur_disturb_r * dur_disturb_choice + \
                               dur_pred / dur_disturb_r * (1 - dur_disturb_choice)
                    dur_pred = torch.round(dur_pred * dur_alpha).clamp(0, 127)
                    dur_pred[:, 0] = 8
                    
                    dur_sum = dur_pred.sum()
                    npad = vqs - dur_sum % vqs
                    if npad < vqs:
                        dur_pred[:, -1] += npad
                    mel2ph_pred = self.length_regulator(dur_pred).to(self.device)

                    # Prepare duration token 
                    print(ph_ref.size(1), mel2ph_ref.max())
                    mel2ph_pred = torch.cat((mel2ph_ref, mel2ph_pred+ph_ref.size(1)), dim=1)
                    mel2ph_pred = mel2ph_pred[:, ::self.vae_stride]

                    sparsified_dur = torch.zeros_like(mel2ph_pred)
                    for i in range(1, mel2ph_pred.max()+1):
                        indices = torch.where(sparsified_dur == i)[0]
                        if len(indices) > 0:
                            rand_idx = indices[torch.randint(len(indices), (1,)).item()]
                            sparsified_dur[rand_idx] = mel2ph_pred[rand_idx]

                    # Disable the English tone (set them to 3)"""
                    ph_pred = torch.cat((ph_ref, ph_pred), dim=1)
                    print(self.ling_dict['phone'].decode(list(ph_pred[0].cpu().numpy())).split(' '))
                    tone_pred = torch.cat((tone_ref, tone_pred), dim=1)
                    en_tone_idx = ~((tone_pred == 4) | ( (11 <= tone_pred) & (tone_pred <= 15)) | (tone_pred == 0))
                    tone_pred[en_tone_idx] = 3
                    ph_seq = torch.cat([ph_pred, ph_pred, torch.full(ph_pred.size(), self.cfg_mask_token_phone, device=self.device)], 0)
                    tone_seq = torch.cat([tone_pred, tone_pred, torch.full(tone_pred.size(), self.cfg_mask_token_tone, device=self.device)], 0)
                    target_size = mel2ph_pred.size(1)
                    vae_latent_ = vae_latent.repeat(3, 1, 1)
                    ctx_mask = torch.ones_like(vae_latent_[:, :, 0:1])
                    vae_latent_ = F.pad(vae_latent_, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
                    ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
                    latent_lengths =  torch.LongTensor([s.shape[0] for s in vae_latent_]).to(self.device)
                    txt_lengths = torch.LongTensor([s.shape[0] for s in ph_seq]).to(self.device)
                    
                    """
                    diffusion model推理所需要的输入：
                    1. 目标文本
                        1. 音
                        2. 语调
                        3. 文本长度
                        4. 预测输出长度
                    2. 提示语音的vae编码图
                    """
                    inputs = {
                        'phone': ph_seq,
                        'tone': tone_seq,
                        'text_lens': txt_lengths,
                        'ctx_mask': ctx_mask,
                        'lat_ctx': vae_latent_ * ctx_mask,
                        'lat': vae_latent_,
                        'lat_lens': latent_lengths,
                        'dur': mel2ph_pred,
                    }

                    with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                        x0 = self.diff_model.inference(inputs, timesteps=32, seq_cfg_w=[3.0, 3.0]).float()
                
                '''
                将diffusion模型的输出转换到另一个空间上
                '''
                # AM
                x0 = x0[:, vae_latent.size(1):]
                mel = self.vocoder.latent2mel(x0.transpose(1, 2))
                # Mute head sil tokens
                mel[:, :, :8] = -6
                # Mute tail sil tokens
                if ph_pred[:, -1] in [148, 153, 166, 145, 163, 165]:
                    tail_sil_len = len(mel2ph_pred[mel2ph_pred==ph_pred.size(1)]) * self.vae_stride
                    mel[:, :, -tail_sil_len:] = -6
                wav_pred = self.vocoder.mel2wav(mel)
                wav_pred = wav_pred.cpu().numpy().astype(float)

                # voc
                yield wav_pred, sr_out, asr_results[0]


def efficient_reference(wav_path, txt_fn, out_path, worker_id, device, infer_ins):
    setproctitle.setproctitle('megatts_inference_worker')
    convert_to_wav(wav_path)
    wav_path = wav_path[:-4] + '.wav'
    os.makedirs(out_path, exist_ok=True)
    try:
        print(f"| Start processing {wav_path}+{txt_fn}")
        inp_txts = [x.strip() for x in open(txt_fn).readlines()]
        inp_txts = [x for x in inp_txts if x != '']
        hooks = []
        # 设置一些参数量
        for block in infer_ins.diff_model.encoder.layers:
            block.attention.step = 0
            block.attention.full_ops = 0
            block.attention.efficient_ops = 0
            # block.attention.need_cache_residual = [True] * len(block.attention.need_cache_residual)
            hook = block.attention.register_forward_pre_hook(calculate_flops_hook, with_kwargs=True)
            hooks.append(hook)

        total_full_ops, total_efficient_ops = 0, 0
        for i, (wav_pred, sr, txt) in enumerate(infer_ins.forward_model([wav_path], inp_txts, out_path)):

            save_wav(wav_pred, f'{out_path}/[Efficient]{inp_txts[i][:20]}.wav', sr=sr)

            # 计算attn ops的变化量以及重置一些参数
            full_ops, efficient_ops = 0, 0
            for block in infer_ins.diff_model.encoder.layers:
                block.attention.step = 0
                full_ops += block.attention.full_ops 
                efficient_ops += block.attention.efficient_ops

                total_full_ops += block.attention.full_ops 
                total_efficient_ops += block.attention.efficient_ops

                block.attention.full_ops = 0
                block.attention.efficient_ops = 0
            print(f"Attn Ops Relative to Full: {round(efficient_ops/full_ops, 4) * 100}")

        # 将一些统计信息写入  

        with open(f"result_{device}.json", 'a+') as write_file:
            from collections import defaultdict
            methodsdict = defaultdict(int)
            for block in infer_ins.diff_model.encoder.layers:
                for method in block.attention.steps_method:
                    methodsdict[method] += 1
            
            method_ratio = []
            for method in ['AST', 'wars+ASC', 'wars', 'ASC', 'full_attention']:
                method_ratio.append(methodsdict[method] / (len(block.attention.steps_method) * len(infer_ins.diff_model.encoder.layers)) )

            write_file.write(json.dumps(
                {
                    'out_path': out_path,
                    'relative ops': round(total_efficient_ops/total_full_ops, 4) * 100,
                    'method_ratio': method_ratio
                }
            ) + "\n")
                
            
        # 移除hooks
        for hook in hooks:
            hook.remove()

    except:
        print(f"| Error occurs when processing {wav_path}+{txt_fn}")
        traceback.print_exc()



def full_reference(wav_path, txt_fn, out_path, worker_id, device, infer_ins):
    setproctitle.setproctitle('megatts_inference_worker')
    for block in infer_ins.diff_model.encoder.layers:
        block.attention.forward = types.MethodType(full_attention_forward, block.attention)

    convert_to_wav(wav_path)
    wav_path = wav_path[:-4] + '.wav'
    os.makedirs(out_path, exist_ok=True)
    try:
        print(f"| Start processing {wav_path}+{txt_fn}")
        inp_txts = [x.strip() for x in open(txt_fn).readlines()]
        inp_txts = [x for x in inp_txts if x != '']
        for i, (wav_pred, sr, txt) in enumerate(infer_ins.forward_model([wav_path], inp_txts, out_path)):
            for block in infer_ins.diff_model.encoder.layers:
                block.attention.step = 0
            save_wav(wav_pred, f'{out_path}/[Full]{inp_txts[i][:20]}.wav', sr=sr)
    except:
        print(f"| Error occurs when processing {wav_path}+{txt_fn}")
        traceback.print_exc()



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

