"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from logging import Logger
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
import copy
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)
from f5_tts.model.logger import Logger


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        self.transformer_uncond = None  # 初始化为None
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    def _create_uncond_transformer(self):
        """创建并设置无条件transformer"""
        if self.transformer_uncond is None:
            self.transformer_uncond = copy.deepcopy(self.transformer)
            # 复制权重
            self.transformer_uncond.load_state_dict(self.transformer.state_dict())
            
            # 设置条件和无条件块的引用关系
            for i, uncond_block in enumerate(self.transformer_uncond.transformer_blocks):
                cond_block = self.transformer.transformer_blocks[i]
                # 确保block_id被正确复制
                if hasattr(cond_block, 'block_id'):
                    uncond_block.block_id = cond_block.block_id
                # 无条件块指向条件块
                uncond_block.cond_dit = cond_block
                # 确保条件块的cond_dit为None
                cond_block.cond_dit = None

    def load_state_dict(self, state_dict, strict=True):
        """重写load_state_dict方法"""
        # 先加载条件模型的权重
        super().load_state_dict(state_dict, strict=strict)
        # 然后创建无条件模型
        self._create_uncond_transformer()

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave
        self._create_uncond_transformer()
        if cond.ndim == 2: 
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(text_lens, lens)  # make sure lengths are at least those of the text characters

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(lens + 1, duration)  # just add one token so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # 设置条件块的引用
        for i, uncond_block in enumerate(self.transformer_uncond.transformer_blocks):
            cond_block = self.transformer.transformer_blocks[i]
            uncond_block.cond_dit = cond_block

        # neural ode

        def fn(t, x, step_cond, text):
            # at each step, conditioning is fixed

            # predict flow
            x, step_cond, text = x.repeat(2, 1, 1), step_cond.repeat(2, 1, 1), text.repeat(2, 1)
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True
            )
            pred, null_pred = pred.chunk(2)
            if cfg_strength < 1e-5:
                return pred

            # null_pred = self.transformer_uncond(
                # x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True
            # )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        self.transformer.set_all_block_id()
        self.transformer_uncond.set_all_block_id()
        #-----------加载策略文件时取消注释--------------
        # self.transformer.load_compression_strategies('method_cond_0.15.json',is_cond=True)
        # self.transformer_uncond.load_compression_strategies('method_uncond_0.15.json',is_cond = False)
        #---------------------------------------------------------
        # trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        trajectory = odeint(
                lambda t, x: fn(t, x, step_cond, text),
                y0,
                t,
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        # self.transformer.save_compression_strategies('method_cond_0.15.json')
        # self.transformer_uncond.save_compression_strategies('method_uncond_0.15.json')

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred

    #! 汇报FLOPS
    def report(self):
        print("CFM.report() called")
        print(f"transformer type: {type(self.transformer)}")
        if isinstance(self.transformer, DiT): 
            print("Transformer is DiT, calling report_flops()")
            flops = self.transformer.report_flops()
            print(f"Got flops: {flops}")
            return flops
        else:
            print(f"Transformer is not DiT: {type(self.transformer)}")
            return None