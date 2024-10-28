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
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP, LlamaRotaryEmbedding, \
    rotate_half, apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

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

