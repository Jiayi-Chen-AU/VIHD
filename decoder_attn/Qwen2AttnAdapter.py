import torch
from torch import nn
from typing import Optional, Tuple

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, repeat_kv, Qwen2RotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

import pdb
from typing import Callable
import wandb
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Qwen2AttnAdapter(Qwen2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int, vision_start=None, vision_end=None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.prefix_len = None
        self.vision_start = None
        self.vision_end = None
        self.prune = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.prefix_len is None:
            self.prefix_len = q_len

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling


        if self.prune and q_len > self.prefix_len: 
            # suppress visual tokens with high attention
            visual_attn = attn_weights[:, :, -1, self.vision_start:self.vision_end].mean(dim=1)
            
            prune_ratio = torch.ones(bsz, device=visual_attn.device) * 0.1
            k_per_sample = (prune_ratio * visual_attn.shape[-1]).long()
            k_per_sample = k_per_sample.clamp(min=1)

            sorted_attn, sorted_idx = torch.sort(visual_attn, dim=-1, descending=True)
            rank = torch.arange(visual_attn.shape[-1], device=visual_attn.device)[None, :]
            rank_mask = rank < k_per_sample[:, None]

            final_mask = torch.zeros_like(visual_attn, dtype=torch.bool, device=visual_attn.device)
            final_mask.scatter_(dim=1, index=sorted_idx, src=rank_mask)

            attn_weights[:, :, -1:, self.vision_start:self.vision_end] = attn_weights[:, :, -1:, self.vision_start:self.vision_end].masked_fill(final_mask[:, None, None, :], float('-1e9'))


        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights