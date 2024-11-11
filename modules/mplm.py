from typing import Optional, Tuple, Union, List

import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from transformers import MistralConfig, Cache
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.mistral.modeling_mistral import (MistralRotaryEmbedding, repeat_kv, MistralMLP,
                                                          MistralRMSNorm, \
    MistralDecoderLayer, MistralAttention, rotate_half)
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP, LlamaRotaryEmbedding, rotate_half, \
    apply_rotary_pos_emb
from torch_geometric.utils import softmax, add_self_loops

import torch.nn.functional as F


def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


class MPLMMistralAttention(MistralAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def forward(self, hidden_states: torch.Tensor, kv_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None, output_attentions: bool = False, use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None, ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        _, k_len, _ = kv_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(kv_states)
        value_states = self.v_proj(kv_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        q_seq_len = q_len
        past_query_length = 0
        past_key_value_length = 0
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index.")
            past_key_value_length = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            kv_seq_len += past_key_value_length
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_query_length = 0
            else:
                past_query_length = past_key_value.get_usable_length(q_seq_len, 0)
            q_seq_len += past_query_length
        cos_q, sin_q = self.rotary_emb(value_states, seq_len=q_seq_len)
        cos_k, sin_k = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states = apply_rotary_pos_emb_single(query_states, cos_q, sin_q,
                                                   torch.arange(past_query_length, q_seq_len,
                                                                device=query_states.device).unsqueeze(0))
        key_states = apply_rotary_pos_emb_single(key_states, cos_k, sin_k,
                                                 torch.arange(past_key_value_length, kv_seq_len,
                                                              device=query_states.device).unsqueeze(0))

        if past_key_value is not None:
            cache_kwargs = {"sin": sin_k, "cos": cos_k}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                             f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                             f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MPLMDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MPLMMistralAttention(config=config, layer_idx=layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, kv_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None, output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False, cache_position: Optional[torch.LongTensor] = None, **kwargs, ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        if kv_states is None:
            kv_states = hidden_states
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        kv_states = self.input_layernorm(kv_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states,
            kv_states=kv_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaRotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_position_embeddings=(128, 2048), base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 4).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device,
            dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = (
        max(seq_len[0], self.max_seq_len_cached[0]), max(seq_len[1], self.max_seq_len_cached[1]))
        x = torch.arange(self.max_seq_len_cached[0], device=device, dtype=self.inv_freq.dtype)
        y = torch.arange(self.max_seq_len_cached[1], device=device, dtype=self.inv_freq.dtype)

        freqs_x = torch.outer(x, self.inv_freq)
        freqs_x = torch.stack([freqs_x] * len(y), dim=1)
        freqs_y = torch.outer(y, self.inv_freq)
        freqs_y = torch.stack([freqs_y] * len(x), dim=0)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs_x, freqs_y, freqs_x, freqs_y), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len[0] > self.max_seq_len_cached[0] or seq_len[1] > self.max_seq_len_cached[1]:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (self.cos_cached[:seq_len[0], :seq_len[1]].to(dtype=x.dtype),
                self.sin_cached[:seq_len[0], :seq_len[1]].to(dtype=x.dtype),)


def apply_rotary_pos_emb_single_2d(q, cos, sin, position_ids, unsqueeze_dim=1):
    rown, coln = position_ids[0].size()[0], position_ids[1].size()[0]
    cos = cos[position_ids[0].view(-1, 1).repeat(1, coln), position_ids[1].repeat(rown, 1)].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids[0].view(-1, 1).repeat(1, coln), position_ids[1].repeat(rown, 1)].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


class MPLMSparseAttention(MessagePassing):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(node_dim=0, aggr="add")
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {self.num_heads}).")
        self.gq_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.gk_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.gv_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.go_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # self.rotary_emb = LlamaRotaryEmbedding2D(
        #     self.head_dim,
        #     max_position_embeddings=(128, 2048),
        #     base=self.rope_theta,
        # )

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=2048, base=self.rope_theta, )

    def forward_llm(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None,
                    output_attentions: bool = False, use_cache: bool = False, **kwargs, ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.gq_proj(hidden_states)
        key_states = self.gk_proj(hidden_states)
        value_states = self.gv_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                             f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                             f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.go_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward(self, hidden_states: torch.Tensor, edge_index: torch.Tensor, edge_hidden_states: torch.Tensor, mem_mask,
            edge_mask, num_nodes, node_order, edge_order, past_key_value, ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.gq_proj(hidden_states)
        key_states = self.gk_proj(hidden_states)
        value_states = self.gv_proj(hidden_states)

        if edge_hidden_states is not None:
            edge_key_states = self.gk_proj(edge_hidden_states)
            edge_value_states = self.gv_proj(edge_hidden_states)
        else:
            edge_key_states = None
            edge_value_states = None

        out = self.propagate(edge_index, query=query_states, key=key_states, value=value_states,
                             edge_key=edge_key_states, edge_value=edge_value_states, mem_mask=mem_mask,
                             edge_mask=edge_mask, num_nodes=num_nodes, node_order=node_order, edge_order=edge_order,
                             past_key_value=past_key_value)

        out = self.go_proj(out)

        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j, edge_key: Tensor, edge_value: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int], mem_mask_j, edge_mask, num_nodes, node_order_i, edge_order,
                past_key_value) -> Tensor:
        bsz, q_len, dim = query_i.size()
        _, k_len, _ = edge_key.size()
        query_states = query_i

        query_pos_ids = torch.ones((bsz, q_len), device=query_states.device, dtype=torch.long)
        query_pos_ids[:, 0] = node_order_i
        query_pos_ids = torch.cumsum(query_pos_ids, dim=-1)
        qmax_pos_ids = query_pos_ids.max() + 1
        overall_mask = torch.cat([mem_mask_j, edge_mask], dim=-1)

        key_pos_ids = overall_mask.clone().to(torch.long)

        key_pos_ids[:, 0] = edge_order
        key_pos_ids = torch.cumsum(key_pos_ids, dim=-1)
        kmax_pos_ids = key_pos_ids.max() + 1

        overall_mask = torch.stack([overall_mask] * q_len, dim=1)
        overall_mask[-num_nodes:, :, q_len:] = 0
        overall_mask = torch.logical_not(overall_mask) * torch.finfo(torch.float32).min
        mask = torch.full((q_len, q_len), torch.finfo(torch.float32).min, device=overall_mask.device)
        mask_cond = torch.arange(mask.size(-1), device=mask.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        overall_mask[-num_nodes:, :, :q_len] += mask
        key_states = torch.cat([key_j, edge_key], dim=1)
        value_states = torch.cat([value_j, edge_value], dim=1)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len + k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len + k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos_q, sin_q = self.rotary_emb(value_states, seq_len=qmax_pos_ids)
        cos_k, sin_k = self.rotary_emb(value_states, seq_len=kmax_pos_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos_q, sin_q, query_pos_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos_k, sin_k, key_pos_ids)
        # key_states = key_states.unsqueeze(0).view(self.num_key_value_heads, bsz, 2* q_len, self.head_dim).permute(
        # 1, 0, 2, 3)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        alpha = (query_states @ key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # overall_mask = torch.logical_not(overall_mask) * torch.finfo(torch.float32).min
        alpha += overall_mask.unsqueeze(1)

        softmax_ind = index.repeat_interleave(q_len + k_len)

        alpha = alpha.permute(1, 0, 3, 2).reshape(self.num_heads, -1, q_len)

        if alpha.size() != (self.num_heads, (q_len + k_len) * bsz, q_len):
            raise ValueError(f"`alpha` should be of size {(self.num_heads, q_len * 2 * bsz, q_len)}, but is"
                             f" {alpha.size()}")

        alpha = softmax(alpha, softmax_ind, num_nodes=size_i, dim=1)
        alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)
        alpha = alpha.view(self.num_heads, bsz, q_len + k_len, q_len).permute(1, 0, 3, 2)

        out = alpha @ value_states
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(bsz, q_len, self.hidden_size)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class MPLMSparseDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MPLMSparseAttention(config=config, layer_idx=layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, edge_index: torch.Tensor, edge_hidden_states: torch.Tensor,
            mem_mask=None, edge_mask=None, num_nodes=None, node_order=None, edge_order=None,
            attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None, output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False, cache_position: Optional[torch.LongTensor] = None, **kwargs, ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        edge_hidden_states = self.input_layernorm(edge_hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states=hidden_states, edge_index=edge_index,
            edge_hidden_states=edge_hidden_states, mem_mask=mem_mask, edge_mask=edge_mask, num_nodes=num_nodes,
            node_order=node_order, edge_order=edge_order, past_key_value=past_key_value)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward_llm(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None,
                    output_attentions: Optional[bool] = False, use_cache: Optional[bool] = False,
                    cache_position: Optional[torch.LongTensor] = None, **kwargs, ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn.forward_llm(hidden_states=hidden_states,
                                                                                         attention_mask=attention_mask,
                                                                                         position_ids=position_ids,
                                                                                         past_key_value=past_key_value,
                                                                                         output_attentions=output_attentions,
                                                                                         use_cache=use_cache,
                                                                                         cache_position=cache_position, )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
