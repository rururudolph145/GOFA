from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, add_self_loops
from transformers import Cache
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP, LlamaRotaryEmbedding, \
    rotate_half, apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, MistralAttention, repeat_kv, \
    MistralMLP, MistralRMSNorm

import pdb
from gp.nn.models.util_model import MLP


class GOFAGNNConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, config):
        super().__init__(node_dim=0, aggr="add")

        self.in_dim = config.dim
        self.in_layer = config.mem_token
        self.dropout = config.dropout
        self.head = config.head

        assert self.in_dim % self.head == 0

        self.d_model = int(self.in_dim / self.head)

        self.add_self_loops = False

        self.lin_qkv = Linear(self.in_dim, self.in_dim * 3, bias=False)

        self.e_proj = Linear(self.in_dim, self.in_dim * 2, bias=False)
        self.layer_norm_ek = LlamaRMSNorm(self.in_dim)
        self.layer_norm_ev = LlamaRMSNorm(self.in_dim)

        self.o_proj = Linear(self.in_dim, self.in_dim, bias=False)

        if config.gnn_mlp_type == "gp":

            self.ff = MLP([self.in_dim, 2 * self.in_dim, self.in_dim], dropout=self.dropout, act=config.gnn_hidden_act)
        elif config.gnn_mlp_type == "llama":
            self.ff = LlamaMLP(config)
        else:
            raise NotImplementedError("Unknown mlp type")

        self.x_norm = LlamaRMSNorm(self.in_dim)
        self.xe_norm = LlamaRMSNorm(self.in_dim)

        self.post_gnn_norm = LlamaRMSNorm(self.in_dim)

        if config.gating:
            self.attn_gate = nn.Parameter(torch.tensor([0.]))
            self.ff_gate = nn.Parameter(torch.tensor([0.]))
        else:
            self.attn_gate = None
            self.ff_gate = None

        if config.position_encoding == "rotary":
            self.rotary_emb = LlamaRotaryEmbedding(self.d_model, max_position_embeddings=self.in_layer,
                                                 base=config.rope_theta, )
        elif config.position_encoding == "none":
            self.rotary_emb = None
        else:
            raise ValueError("Unknown position encoding.")

        self.mp_att = config.mp_att

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, xe: Tensor):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
        """
        x = x.view(x.size()[0], self.in_layer, self.in_dim)
        # Q = x.clone().detach()
        residual = x
        x = self.x_norm(x)
        xe = xe.view(xe.size()[0], self.in_layer, self.in_dim)
        xe = self.xe_norm(xe)

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, xe = add_self_loops(edge_index, edge_attr=xe, fill_value=0.0, num_nodes=num_nodes)

        qkv = self.lin_qkv(x)
        query, key, value = torch.chunk(qkv, 3, -1)

        xe = self.e_proj(xe)

        xe_key, xe_value = torch.chunk(xe, 2, -1)

        out = self.propagate(edge_index, query=query, key=key, value=value, xe_key=xe_key, xe_value=xe_value)
        out = self.o_proj(out)
        if self.attn_gate is not None:
            out = residual + out * self.attn_gate.tanh()
        else:
            out = residual + out

        residual = out

        out = self.post_gnn_norm(out)
        out = self.ff(out)
        if self.ff_gate is not None:
            out = residual + out * self.ff_gate.tanh()
        else:
            out = residual + out
        #
        # H = out.clone().detach().view(Q.size())
        # diff = (((H - Q).to(torch.float32))**2).sum(dim=(-2))/((Q.to(torch.float32))**2).sum(dim=(-2))
        # print(diff.sum(), len(diff.view(-1)), diff.mean())

        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor, xe_key: Tensor, xe_value: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        key_j = xe_key + key_j
        value_j = value_j + xe_value
        query_i = query_i.view(-1, self.in_layer, self.head, self.d_model)
        key_j = key_j.view(-1, self.in_layer, self.head, self.d_model)
        value_j = value_j.view(-1, self.in_layer, self.head, self.d_model)
        if self.rotary_emb is not None:
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
            cos, sin = self.rotary_emb(value_j, self.in_layer)
            query_i, key_j = apply_rotary_pos_emb(query_i, key_j, cos, sin, torch.arange(self.in_layer, device=value_j.device).unsqueeze(0))
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.d_model)

        alpha = softmax(alpha, index, ptr, size_i, dim=0)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j.view(-1, self.d_model)
        out = out * alpha.view(-1, 1)
        out = out.view(-1, self.in_layer, self.in_dim)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GOFAGNNConvFullAtt(GOFAGNNConv):
    def __init__(self, config):
        super().__init__(config)

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor, xe_key: Tensor, xe_value: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        key_j = key_j
        value_j = value_j
        query_i = query_i.view(-1, self.in_layer, self.head, self.d_model)
        key_j = key_j.view(-1, self.in_layer, self.head, self.d_model)
        value_j = value_j.view(-1, self.in_layer, self.head, self.d_model)
        if self.rotary_emb is not None:
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
            cos, sin = self.rotary_emb(value_j, self.in_layer)
            query_i, key_j = apply_rotary_pos_emb(query_i, key_j, cos, sin, torch.arange(self.in_layer, device=value_j.device).unsqueeze(0))
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
        key_j = key_j.permute(2, 0, 1, 3)
        query_i = query_i.permute(2, 0, 3, 1)
        value_j = value_j.permute(2, 0, 1, 3)
        alpha = (key_j @ query_i) / math.sqrt(self.d_model)
        softmax_ind = index.repeat_interleave(self.in_layer)

        alpha = alpha.view(self.head, -1, self.in_layer)

        alpha = softmax(alpha, softmax_ind, num_nodes=size_i, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha = alpha.view(self.head, -1, self.in_layer, self.in_layer).transpose(-1, -2)

        out = alpha @ value_j
        out = out.permute(1, 2, 0, 3).reshape(-1, self.in_layer, self.in_dim)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_position_embeddings=(2048, 2048), base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class GOFAGNNConvCMB(MessagePassing):
    _alpha: OptTensor

    def __init__(self, config):
        super().__init__(node_dim=0, aggr="add")

        self.in_dim = config.dim
        self.in_layer = config.mem_token
        self.dropout = config.dropout
        self.head = config.head

        assert self.in_dim % self.head == 0

        self.d_model = int(self.in_dim / self.head)

        self.add_self_loops = False

        self.nq_proj = Linear(self.in_dim, self.in_dim, bias=False)
        self.nk_proj = Linear(self.in_dim, self.in_dim, bias=False)
        self.nv_proj = Linear(self.in_dim, self.in_dim, bias=False)

        self.ek_proj = Linear(self.in_dim, self.in_dim, bias=False)
        self.ev_proj = Linear(self.in_dim, self.in_dim, bias=False)
        self.layer_norm_ek = LlamaRMSNorm(self.in_dim)
        self.layer_norm_ev = LlamaRMSNorm(self.in_dim)

        self.o_proj = Linear(self.in_dim, self.in_dim, bias=False)

        if config.gnn_mlp_type == "gp":

            self.ff = MLP([self.in_dim, 2 * self.in_dim, self.in_dim], dropout=self.dropout, act=config.gnn_hidden_act)
        elif config.gnn_mlp_type == "llama":
            self.ff = LlamaMLP(config)
        else:
            raise NotImplementedError("Unknown mlp type")

        self.x_norm = LlamaRMSNorm(self.in_dim)
        self.xe_norm = LlamaRMSNorm(self.in_dim)

        self.post_gnn_norm = LlamaRMSNorm(self.in_dim)

        if config.gating:
            self.attn_gate = nn.Parameter(torch.tensor([0.]))
            self.ff_gate = nn.Parameter(torch.tensor([0.]))
        else:
            self.attn_gate = None
            self.ff_gate = None

        if config.position_encoding == "rotary":
            self.rotary_emb = LlamaRotaryEmbedding(self.d_model, max_position_embeddings=self.in_layer,
                                                 base=config.rope_theta, )
        elif config.position_encoding == "none":
            self.rotary_emb = None
        else:
            raise ValueError("Unknown position encoding.")

        self.mp_att = config.mp_att

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, xe: Tensor):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
        """
        x = x.view(x.size()[0], self.in_layer, self.in_dim)
        # Q = x.clone().detach()
        residual = x
        x = self.x_norm(x)
        xe = xe.view(xe.size()[0], self.in_layer, self.in_dim)
        xe = self.xe_norm(xe)

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, xe = add_self_loops(edge_index, edge_attr=xe, fill_value=0.0, num_nodes=num_nodes)

        query = self.nq_proj(x)

        xe_key, xe_value = self.ek_proj(xe), self.ev_proj(xe)

        out = self.propagate(edge_index, query=query, x=x, xe_key=xe_key, xe_value=xe_value)
        out = self.o_proj(out)
        if self.attn_gate is not None:
            out = residual + out * self.attn_gate.tanh()
        else:
            out = residual + out

        residual = out

        out = self.post_gnn_norm(out)
        out = self.ff(out)
        if self.ff_gate is not None:
            out = residual + out * self.ff_gate.tanh()
        else:
            out = residual + out
        #
        # H = out.clone().detach().view(Q.size())
        # diff = (((H - Q).to(torch.float32))**2).sum(dim=(-2))/((Q.to(torch.float32))**2).sum(dim=(-2))
        # print(diff.sum(), len(diff.view(-1)), diff.mean())

        return out

    def message(self, query_i: Tensor, x_j: Tensor, xe_key: Tensor, xe_value: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        key_j = self.nk_proj(xe_key + x_j)
        value_j = self.nv_proj(xe_value + x_j)
        query_i = query_i.view(-1, self.in_layer, self.head, self.d_model)
        key_j = key_j.view(-1, self.in_layer, self.head, self.d_model)
        value_j = value_j.view(-1, self.in_layer, self.head, self.d_model)
        if self.rotary_emb is not None:
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
            cos, sin = self.rotary_emb(value_j, self.in_layer)
            query_i, key_j = apply_rotary_pos_emb(query_i, key_j, cos, sin, torch.arange(self.in_layer, device=value_j.device).unsqueeze(0))
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
        key_j = key_j.permute(2, 0, 1, 3)
        query_i = query_i.permute(2, 0, 3, 1)
        value_j = value_j.permute(2, 0, 1, 3)
        alpha = (key_j @ query_i) / math.sqrt(self.d_model)
        softmax_ind = index.repeat_interleave(self.in_layer)

        alpha = alpha.view(self.head, -1, self.in_layer)

        alpha = softmax(alpha, softmax_ind, num_nodes=size_i, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha = alpha.view(self.head, -1, self.in_layer, self.in_layer).transpose(-1, -2)

        out = alpha @ value_j
        out = out.permute(1, 2, 0, 3).reshape(-1, self.in_layer, self.in_dim)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class SMMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, last_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        if last_size is None:
            self.last_size = self.hidden_size
        else:
            self.last_size = last_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.last_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class GOFAGNNConvMLP(MessagePassing):
    _alpha: OptTensor

    def __init__(self, config):
        super().__init__(node_dim=0, aggr="add")

        self.in_dim = config.dim
        self.in_layer = config.mem_token
        self.dropout = config.dropout
        self.head = config.head

        assert self.in_dim % self.head == 0

        self.d_model = int(self.in_dim / self.head)

        self.add_self_loops = False

        self.nq_proj = Linear(self.in_dim, self.in_dim, bias=False)
        self.nk_proj = Linear(self.in_dim, self.in_dim, bias=False)
        self.nv_proj = Linear(self.in_dim, self.in_dim, bias=False)

        self.layer_norm_ek = LlamaRMSNorm(self.in_dim)
        self.layer_norm_ev = LlamaRMSNorm(self.in_dim)

        self.o_proj = Linear(self.in_dim, self.in_dim, bias=False)

        self.mixer_mlp = SMMLP(self.in_layer*2, self.in_layer*4, self.in_layer)

        if config.gnn_mlp_type == "gp":

            self.ff = MLP([self.in_dim, 2 * self.in_dim, self.in_dim], dropout=self.dropout, act=config.gnn_hidden_act)
        elif config.gnn_mlp_type == "llama":
            self.ff = LlamaMLP(config)
        else:
            raise NotImplementedError("Unknown mlp type")

        self.x_norm = LlamaRMSNorm(self.in_dim)
        self.xe_norm = LlamaRMSNorm(self.in_dim)

        self.post_gnn_norm = LlamaRMSNorm(self.in_dim)

        if config.gating:
            self.attn_gate = nn.Parameter(torch.tensor([0.]))
            self.ff_gate = nn.Parameter(torch.tensor([0.]))
        else:
            self.attn_gate = None
            self.ff_gate = None

        if config.position_encoding == "rotary":
            self.rotary_emb = LlamaRotaryEmbedding(self.d_model, max_position_embeddings=self.in_layer,
                                                 base=config.rope_theta, )
        elif config.position_encoding == "none":
            self.rotary_emb = None
        else:
            raise ValueError("Unknown position encoding.")

        self.mp_att = config.mp_att

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, xe: Tensor):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
        """
        x = x.view(x.size()[0], self.in_layer, self.in_dim)
        # Q = x.clone().detach()
        residual = x
        x = self.x_norm(x)
        xe = xe.view(xe.size()[0], self.in_layer, self.in_dim)
        xe = self.xe_norm(xe)

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, xe = add_self_loops(edge_index, edge_attr=xe, fill_value=0.0, num_nodes=num_nodes)

        query = self.nq_proj(x)

        out = self.propagate(edge_index, query=query, x=x, xe=xe)
        out = self.o_proj(out)
        if self.attn_gate is not None:
            out = residual + out * self.attn_gate.tanh()
        else:
            out = residual + out

        residual = out

        out = self.post_gnn_norm(out)
        out = self.ff(out)
        if self.ff_gate is not None:
            out = residual + out * self.ff_gate.tanh()
        else:
            out = residual + out
        #
        # H = out.clone().detach().view(Q.size())
        # diff = (((H - Q).to(torch.float32))**2).sum(dim=(-2))/((Q.to(torch.float32))**2).sum(dim=(-2))
        # print(diff.sum(), len(diff.view(-1)), diff.mean())

        return out

    def message(self, query_i: Tensor, x_j: Tensor, xe: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        ne_emb = self.mixer_mlp(torch.cat([x_j, xe], dim=-2).transpose(-1, -2)).transpose(-1, -2)
        key_j = self.nk_proj(ne_emb)
        value_j = self.nv_proj(ne_emb)
        query_i = query_i.view(-1, self.in_layer, self.head, self.d_model)
        key_j = key_j.view(-1, self.in_layer, self.head, self.d_model)
        value_j = value_j.view(-1, self.in_layer, self.head, self.d_model)
        if self.rotary_emb is not None:
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
            cos, sin = self.rotary_emb(value_j, self.in_layer)
            query_i, key_j = apply_rotary_pos_emb(query_i, key_j, cos, sin, torch.arange(self.in_layer, device=value_j.device).unsqueeze(0))
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
        key_j = key_j.permute(2, 0, 1, 3)
        query_i = query_i.permute(2, 0, 3, 1)
        value_j = value_j.permute(2, 0, 1, 3)
        alpha = (key_j @ query_i) / math.sqrt(self.d_model)
        softmax_ind = index.repeat_interleave(self.in_layer)

        alpha = alpha.view(self.head, -1, self.in_layer)

        alpha = softmax(alpha, softmax_ind, num_nodes=size_i, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha = alpha.view(self.head, -1, self.in_layer, self.in_layer).transpose(-1, -2)

        out = alpha @ value_j
        out = out.permute(1, 2, 0, 3).reshape(-1, self.in_layer, self.in_dim)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GOFAGNN(torch.nn.Module):
    def __init__(self, gofa_config):
        super().__init__()
        if gofa_config.gnn_type == "index":
            self.g_layers = nn.ModuleList([GOFAGNNConv(gofa_config) for _ in range(gofa_config.mid_num_layers)])
        elif gofa_config.gnn_type == "full":
            self.g_layers = nn.ModuleList([GOFAGNNConvFullAtt(gofa_config) for _ in range(gofa_config.mid_num_layers)])
        else:
            raise ValueError("Unknown GNN type for GOFA.")
        self.mem_token = gofa_config.mem_token

    def forward(self, hidden_states, graph, map_node=None):
        cur_node_size = graph.num_node_feat
        for g_layer_idx in range(len(self.g_layers)):
            if g_layer_idx == 0 and map_node:
                hidden_states = torch.cat(
                    [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]], dim=0)
                cur_node_size = len(graph.node_map)
            mem_repr = hidden_states.view(hidden_states.size()[0], self.mem_token, -1)
            gnn_input = mem_repr[:cur_node_size]
            gnn_edge_input = mem_repr[cur_node_size:][graph.edge_map]

            output = self.g_layers[g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input)
            output = torch.cat([output, mem_repr[cur_node_size:]], dim=0)
            hidden_states = output
        return hidden_states

def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


class GOFAAttention(MessagePassing):
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
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.gq_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.gk_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.gv_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.go_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden_states: torch.Tensor,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.gq_proj(hidden_states)
        key_states = self.gk_proj(hidden_states)
        value_states = self.gv_proj(hidden_states)

        edge_key_states = self.gk_proj(edge_hidden_states)
        edge_value_states = self.gv_proj(edge_hidden_states)

        out = self.propagate(edge_index, query=query_states, key=key_states, value=value_states, edge_key=edge_key_states, edge_value=edge_value_states)

        out = self.go_proj(out)

        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j, edge_key: Tensor, edge_value: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        bsz, q_len, dim = query_i.size()
        query_states = query_i
        key_states = torch.stack([key_j, edge_key], dim=2).view(bsz, q_len*2, -1)
        value_states = torch.stack([value_j, edge_value], dim=2).view(bsz, q_len*2, -1)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, 2*q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, 2*q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos_q, sin_q = self.rotary_emb(value_states, seq_len=q_len)
        cos_k, sin_k = self.rotary_emb(value_states, seq_len=q_len*2)
        query_states = apply_rotary_pos_emb_single(query_states, cos_q, sin_q, torch.arange(0, q_len, device=query_states.device).unsqueeze(0))
        key_states = apply_rotary_pos_emb_single(key_states, cos_k, sin_k, torch.arange(0, q_len*2, device=query_states.device).unsqueeze(0))

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        alpha = (query_states @ key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        softmax_ind = index.repeat_interleave(q_len*2)

        alpha = alpha.permute(1, 0, 3, 2).reshape(self.num_heads, -1, q_len)

        if alpha.size() != (self.num_heads, q_len * 2 * bsz, q_len):
            raise ValueError(f"`alpha` should be of size {(self.num_heads, q_len * 2 * bsz, q_len)}, but is"
                             f" {alpha.size()}")

        alpha = softmax(alpha, softmax_ind, num_nodes=size_i, dim=1)
        alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)
        alpha = alpha.view(self.num_heads, bsz, q_len*2, q_len).permute(1, 0, 3, 2)

        out = alpha @ value_states
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(bsz, q_len, self.hidden_size)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GOFADecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GOFAAttention(config=config, layer_idx=layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden_states: torch.Tensor,
    ):
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
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            edge_index=edge_index,
            edge_hidden_states=edge_hidden_states,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
