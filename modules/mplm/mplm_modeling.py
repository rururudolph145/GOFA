import math
from collections import OrderedDict
from typing import Optional, List, Union, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from transformers import MistralModel, MistralConfig, MistralForCausalLM, AutoTokenizer, Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, rotate_half, repeat_kv, MistralMLP

from modules.gofa.gofa_modeling import logger


def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


class GOFACache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.edge_cache = {}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_usable_length(self, seq_len, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_edge_states(self, layer_index):
        if layer_index not in self.edge_cache:
            return None
        return self.edge_cache[layer_index]

    def update_edge_states(self, key_cache, value_cache, layer_index):
        self.edge_cache[layer_index] = (key_cache, value_cache)

    def remove_edge(self, n_node):
        for i in range(len(self.key_cache) - len(self.edge_cache)):
            self.key_cache[i] = self.key_cache[i][:n_node]
            self.value_cache[i] = self.value_cache[i][:n_node]


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

        if past_key_value is None or past_key_value.get_edge_states(self.layer_idx) is None:
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
        query_states = query_i

        n_max_token_size = mem_mask_j.size()[-1]
        e_max_token_size = edge_mask.size()[-1]

        past_q_len = 0
        if past_key_value is not None:
            past_q_len += past_key_value.get_usable_length(q_len, self.layer_idx)

        query_pos_ids = mem_mask_j.clone().to(torch.long)
        query_pos_ids[:, 0] += node_order_i - 1
        query_pos_ids = torch.cumsum(query_pos_ids, dim=-1)
        query_pos_ids = query_pos_ids[:, past_q_len:]
        qmax_pos_ids = query_pos_ids.max() + 1

        # TODO: This cause problem when there are multiple generation point in generation mode, fine in training.
        overall_mask = torch.cat([mem_mask_j, edge_mask], dim=-1)

        key_pos_ids = overall_mask.clone().to(torch.long)

        key_pos_ids[:, 0] += edge_order - 1
        key_pos_ids = torch.cumsum(key_pos_ids, dim=-1)
        kmax_pos_ids = key_pos_ids.max() + 1
        key_node_pos_ids, key_edge_pos_ids = key_pos_ids[:, : n_max_token_size], key_pos_ids[:, n_max_token_size:]
        key_node_pos_ids = key_node_pos_ids[:, past_q_len:]
        overall_mask = torch.stack([overall_mask] * q_len, dim=1)
        overall_mask[-num_nodes:, :, n_max_token_size:] = 0
        overall_mask = torch.logical_not(overall_mask) * torch.finfo(torch.float32).min
        mask = torch.full((q_len + past_q_len, q_len + past_q_len), torch.finfo(torch.float32).min, device=overall_mask.device)
        mask_cond = torch.arange(mask.size(-1), device=mask.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        overall_mask[-num_nodes:, :, :q_len + past_q_len] += mask[-q_len:,]

        # key_states = torch.cat([key_j, edge_key], dim=1)
        # value_states = torch.cat([value_j, edge_value], dim=1)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_j.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_j.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos_q, sin_q = self.rotary_emb(value_states, seq_len=qmax_pos_ids)
        cos_k, sin_k = self.rotary_emb(value_states, seq_len=kmax_pos_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos_q, sin_q, query_pos_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos_k, sin_k, key_node_pos_ids)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        if past_key_value is None or past_key_value.get_edge_states(self.layer_idx) is None:
            key_edge_states = edge_key.view(bsz, e_max_token_size, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_edge_states = edge_value.view(bsz, e_max_token_size, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            key_edge_states = apply_rotary_pos_emb_single(key_edge_states, cos_k, sin_k, key_edge_pos_ids)
            if past_key_value is not None:
                past_key_value.update_edge_states(key_edge_states, value_edge_states, self.layer_idx)
        else:
            key_edge_states, value_edge_states = past_key_value.get_edge_states(self.layer_idx)
        key_states = torch.cat([key_states, key_edge_states], dim=-2)
        value_states = torch.cat([value_states, value_edge_states], dim=-2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        alpha = (query_states @ key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # overall_mask = torch.logical_not(overall_mask) * torch.finfo(torch.float32).min
        alpha += overall_mask.unsqueeze(1)

        softmax_ind = index.repeat_interleave(past_q_len + q_len + e_max_token_size)

        alpha = alpha.permute(1, 0, 3, 2).reshape(self.num_heads, -1, q_len)

        if alpha.size() != (self.num_heads, (past_q_len + q_len + e_max_token_size) * bsz, q_len):
            raise ValueError(f"`alpha` should be of size {(self.num_heads, q_len * 2 * bsz, q_len)}, but is"
                             f" {alpha.size()}")

        alpha = softmax(alpha, softmax_ind, num_nodes=size_i, dim=1)
        alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)
        alpha = alpha.view(self.num_heads, bsz, past_q_len + q_len + e_max_token_size, q_len).permute(1, 0, 3, 2)
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
            past_key_value: Optional[Cache] = None, use_cache: Optional[bool] = False, **kwargs, ):
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


class MPLMSparseModel(MistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: MistralConfig, gofa_config):
        super().__init__(config)
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.g_layers = nn.ModuleList()
        self.g_layers.append(nn.ModuleList([MPLMSparseDecoderLayer(config, i+len(self.layers)- gofa_config.num_layers) for i in range(gofa_config.num_layers)]))
        self.g_layers.append(MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        # self.g_layers.append(nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for i in range(gofa_config.num_layers)]))
        self.mem_token = gofa_config.mem_token
        self.llama_dtype = gofa_config.llama_dtype
        self.n_layers = gofa_config.num_layers
        self.trainable_layer = gofa_config.trainable_layer

    def align_weight(self):
        n_layers = len(self.layers)
        inactive_layers = n_layers - len(self.g_layers[0])
        partial_state_dict = OrderedDict()
        source_dict = self.layers.state_dict()
        for layer_name in source_dict:
            name_split = layer_name.split(".")
            layer_ind = int(name_split[0])
            if layer_ind >= inactive_layers:
                name_split[0] = str(layer_ind - inactive_layers)
                if name_split[2] in ["v_proj", "q_proj", "k_proj", "o_proj"]:
                    name_split[2] = "g"+name_split[2]
                partial_state_dict[".".join(name_split)] = source_dict[layer_name]

        self.g_layers[0].load_state_dict(partial_state_dict)
        self.g_layers[1].load_state_dict(self.norm.state_dict())
        self.layers = self.layers[:inactive_layers]
        # for layer in self.g_layers[2]:
        #     layer.weight.data.copy_(torch.eye(len(layer.weight.data)))

    def set_trainable_state(self, lora=False):
        self.layers.requires_grad_(False)
        for i in range(len(self.g_layers[0]) - self.trainable_layer):
            self.g_layers[0][i].requires_grad_(False)
        # self.g_layers[2].requires_grad_(True)
        # if lora:
        #     for i in range(len(self.g_layers[0]) - self.trainable_layer, len(self.g_layers[0])):
        #         for name, submodule in self.g_layers[0][i].named_modules():
        #             if isinstance(submodule, MistralRMSNorm):
        #                 for param in submodule.parameters():
        #                     param.requires_grad = True
        #     self.g_layers[1].requires_grad_(True)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            graph=None,
            mem_mask=None,
            edge_mask=None,
            partial_grad=None,
            map_node=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        past_key_state = past_key_values is not None

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            raise ValueError("You cannot specify input_ids for GOFA, please construct input embeddings manually")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = GOFACache()
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            # attention_mask = _prepare_4d_causal_attention_mask(
            #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
            # )
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, sliding_window=self.config.sliding_window,
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None


        cur_node_size = graph.num_node_feat if graph is not None else 0
        for i in range(len(self.layers) + len(self.g_layers[0])):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            g_layer_idx = i - len(self.layers)
            if g_layer_idx < 0:
                decoder_layer = self.layers[i]
            if g_layer_idx == 0 and map_node:
                hidden_states = torch.cat(
                    [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]], dim=0)
                node_mask = mem_mask[graph.node_map]
                edge_mask = edge_mask[graph.edge_map]
                attention_mask = torch.cat(
                    [attention_mask[:cur_node_size][graph.node_map], attention_mask[cur_node_size:]], dim=0)
                cur_node_size = len(graph.node_map)
            if g_layer_idx >= 0 and graph is not None:
                # hidden_states = self.g_layers[2][g_layer_idx](hidden_states)
                gnn_input = hidden_states[:cur_node_size]
                if past_key_values is None:
                    gnn_edge_input = hidden_states[cur_node_size:][:, :graph.max_edge_token][graph.edge_map]
                elif past_key_values.get_edge_states(i) is None:
                    gnn_edge_input = hidden_states[cur_node_size:][:, :graph.max_edge_token][graph.edge_map]
                else:
                    gnn_edge_input = hidden_states[cur_node_size:]

                graph_output = self.g_layers[0][g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input, mem_mask=node_mask, edge_mask=edge_mask, num_nodes=graph.num_nodes, node_order=graph.node_order, edge_order=graph.edge_order, past_key_value=past_key_values)
            else:
                graph_output = None
            if g_layer_idx < 0 and partial_grad:
                with torch.no_grad():
                    layer_outputs = self.forward_llm_layer(hidden_states, self.llama_dtype, decoder_layer, attention_mask, position_ids, past_key_values, output_attentions, use_cache)
            elif len(hidden_states) > cur_node_size and (i < len(self.layers) + len(self.g_layers[0]) - self.trainable_layer):
                with torch.no_grad():
                    layer_outputs = self.g_layers[0][g_layer_idx].forward_llm(hidden_states[cur_node_size:],
                                                                              attention_mask=attention_mask[
                                                                                             cur_node_size:],
                                                                              position_ids=position_ids,
                                                                              output_attentions=output_attentions,
                                                                              use_cache=use_cache, )
            elif len(hidden_states) > cur_node_size:
                layer_outputs = self.g_layers[0][g_layer_idx].forward_llm(hidden_states[cur_node_size:],
                                                                          attention_mask=attention_mask[cur_node_size:],
                                                                          position_ids=position_ids,
                                                                          output_attentions=output_attentions,
                                                                          use_cache=use_cache, )
            else:
                layer_outputs = (None, None)
                # layer_outputs = decoder_layer(hidden_states[cur_node_size:], attention_mask=attention_mask[cur_node_size:], position_ids=position_ids,
                #                               past_key_value=past_key_values, output_attentions=output_attentions,
                #                               use_cache=use_cache, )

            hidden_states = layer_outputs[0]
            if graph_output is not None and hidden_states is not None:
                hidden_states = torch.cat([graph_output, hidden_states], dim=0)
            elif hidden_states is None:
                hidden_states = graph_output

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        if graph is not None:
            hidden_states = self.g_layers[1](hidden_states)
        else:
            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = past_key_values
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def forward_llm_layer(self, hidden_states, dtype, decoder_layer, attention_mask, position_ids, past_key_values, output_attentions, use_cache):
        hidden_states = hidden_states.to(dtype)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask,
                position_ids, past_key_values, output_attentions, use_cache, )
        else:
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_values, output_attentions=output_attentions, use_cache=use_cache, )

        return layer_outputs


class MPLMSparseForCausalLM(MistralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, gofa_config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = MPLMSparseModel(config, gofa_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        graph = None,
        mem_mask = None,
        edge_mask = None,
        partial_grad = None,
        map_node = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            graph=graph,
            mem_mask=mem_mask,
            partial_grad=partial_grad,
            map_node=map_node,
            edge_mask=edge_mask,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_base_model(self):
        return self
