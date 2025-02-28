"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations


import time
from time import perf_counter
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
import os
import json
import copy
import wandb

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from tqdm import tqdm

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
	default,
	exists,
	lens_to_mask,
	list_str_to_idx,
	list_str_to_tensor,
	mask_from_frac_lengths
)
from f5_tts.model.modules import AttnProcessor

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
		dim = transformer.dim
		self.dim = dim

		# conditional flow related
		self.sigma = sigma

		# sampling related
		self.odeint_kwargs = odeint_kwargs

		# vocab map for tokenization
		self.vocab_char_map = vocab_char_map
		
		# Initialize wandb in offline mode
		wandb.init(project="f5_tts_compression", mode="offline", dir="./wandb_logs")
		
	def load_state_dict(self, state_dict, strict=True):
		"""重写load_state_dict方法"""
		# 先加载条件模型的权重
		super().load_state_dict(state_dict, strict=strict)

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
		delta = None,
		timer = False
	):
		self.eval()
  
		# raw wave
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

		# neural ode

		def fn(t, x, step_cond, text):
			# at each step, conditioning is fixed
			# step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

			# predict flow
			x, step_cond, text = x.repeat(2, 1, 1), step_cond.repeat(2, 1, 1), text.repeat(2, 1)
			pred = self.transformer(
				x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True
			)
			pred, null_pred = pred.chunk(2)
			if cfg_strength < 1e-5:
				return pred

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

		return out, trajectory


	@torch.no_grad()
	def eval_method(
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
		delta = None,
		timer = False,
	):
		self.eval()
  
		# raw wave
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

		# neural ode

		def fn(t, x, step_cond, text):
			# at each step, conditioning is fixed
			# step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

			# predict flow
			x, step_cond, text = x.repeat(2, 1, 1), step_cond.repeat(2, 1, 1), text.repeat(2, 1)
			pred = self.transformer(
				x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True
			)
			pred, null_pred = pred.chunk(2)
			if cfg_strength < 1e-5:
				return pred

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
		odeint(
				lambda t, x: fn(t, x, step_cond, text),
				y0,
				t,
				atol=1e-4,
				rtol=1e-4,
				method="euler",
		)
  
		def calculate_total_cost(transformer):
			METHOD_COST_UNIT = {
				'full_attention': 0.5,
				'ast': 0,
				'asc': 0,
				'wars': 0.0625
			}
			total_cost = 0
			method_static = {
				'full_attention': 0,
				'ast': 0,
				'asc': 0,
				'wars': 0
			}
			# 统计每个块和步骤的策略计算量
			for blocki, block in enumerate(transformer.transformer_blocks):
				for stepi in range(len(block.attn.steps_method)):
					method = block.attn.steps_method[stepi]
					for i in range(2):
						if method[i] in METHOD_COST_UNIT:
							total_cost += METHOD_COST_UNIT[method[i]] * (1.125 if block.attn.need_cache_residual[stepi][i] and method[i] == 'full_attention' else 1.0)
							method_static[method[i]] += 1
						else:
							raise ValueError(f"Unknown method: {method}")
			
			print(method_static)
			
			return total_cost

		return calculate_total_cost(self.transformer)


	# 校准
	@torch.no_grad()
	def calibrate(self,
		cond: float["b n d"] | float["b nw"],  # noqa: F722
		text: int["b nt"] | list[str],  # noqa: F722
		duration: int | int["b"],  # noqa: F821
		*,
		lens: int["b"] | None = None,  # noqa: F821
		steps=2,
		sway_sampling_coef=None,
		seed: int | None = None,
		max_duration=4096,
		no_ref_audio=False,
		edit_mask=None,
		delta = 0.1 # 压缩阈值
	):
		self.eval()
		# raw wave
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

		y0 = []
		for dur in duration:
			if exists(seed):
				torch.manual_seed(seed)
			y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
		y0 = pad_sequence(y0, padding_value=0, batch_first=True)

		t_start = 0

		t = torch.linspace(t_start, 1, steps, device=self.device, dtype=step_cond.dtype)
		if sway_sampling_coef is not None:
			t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

		self.transformer.set_all_block_id()

		# 复制输出
		y0_rep = y0.repeat(2, 1, 1)
		step_cond_rep = step_cond.repeat(2, 1, 1)
		text_rep = text.repeat(2, 1)

		# 遍历各个时间步和各个块，挑选方法，首先初始化压缩策略
		method_dict = {
			str(block_id): {
				f"{t[i].item():.3f}": "none" for i in range(len(t))
			} for block_id in range(len(self.transformer.transformer_blocks))
		}
		none_method_dict = copy.deepcopy(method_dict) # 重置策略
		method_candidate = ['ast','none']
		# 先收集所有时间步的无压缩输出
		output_nospeedup = {}
		
		for t_step in t[1:]: # 第一个时间步总是none
			self.transformer.load_compression_strategies(none_method_dict)
			self._reset_compress_manager()
			output_nospeedup[f'{t_step.item():.3f}'] = self.transformer(
				x=y0_rep, cond=step_cond_rep, text=text_rep, time=t_step, mask=mask, 
				drop_audio_cond=True, drop_text=True
			)
		
		# 对第一个时间步进行缓存，使得第二个时间步有机会用ast
		self.transformer.set_all_block_need_cal_window_res()
		self.transformer(
				x=y0_rep, cond=step_cond_rep, text=text_rep, time=t[0], mask=mask, 
				drop_audio_cond=True, drop_text=True
			)
		
		# 遍历各种策略
		total_steps = len(t[1:]) * len(self.transformer.transformer_blocks) * len(method_candidate)
		with tqdm(total=total_steps, desc="Searching compression strategies") as pbar:
			for block_id in range(len(self.transformer.transformer_blocks)):
				for t_step in t[1:]: # 第一个时间步总是none
					for method in method_candidate:
						# 更新进度条描述
						pbar.set_description(f"t={t_step.item():.3f}, block={block_id}, trying {method}")
						
						# 尝试方法
						method_dict[str(block_id)][f'{t_step.item():.3f}'] = method
						self.transformer.load_compression_strategies(method_dict)
						self.transformer.set_all_block_need_cal_window_res()
						# 当前输出
						pred_speedup = self.transformer(
								x=y0_rep, cond=step_cond_rep, text=text_rep, time=t_step, mask=mask, drop_audio_cond=True, drop_text=True
						)
						# 计算比较结果
						compare_result = self._compression_compare(output_nospeedup[f'{t_step.item():.3f}'], pred_speedup)
						# 记录日志到wandb
						wandb.log({
							"time_step": t_step.item(),
							"block_id": block_id, 
							"method": method,
							"diff": compare_result,
							"threshold": delta * (block_id+1) / len(self.transformer.transformer_blocks)
						})
						if compare_result < delta * (block_id+1) / len(self.transformer.transformer_blocks):
							pbar.update(len(method_candidate) - method_candidate.index(method))  # 跳过剩余的方法
							break
						pbar.update(1)
					else:
						method_dict[str(block_id)][f'{t_step.item():.3f}'] = 'none'
						
		# 保存策略
		with open(os.path.join('method' + str(delta) + '.json'), "w") as f:
			json.dump(method_dict, f, indent=4)
			
		print("策略保存成功")

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