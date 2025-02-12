import torch
from transformers import LlamaConfig, MistralConfig



class GOFALlamaConfig(LlamaConfig):
    def __init__(self, dim=4096, num_layers=6, mem_token=128, head=32, add_self_loops=True, dropout=0.0,
                 llama_dtype=torch.float16, gnn_hidden_act="relu", gnn_mlp_type="gp", pretraining_tp=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mem_token = mem_token
        self.head = head
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.num_layers = num_layers
        self.llama_dtype = llama_dtype
        self.gnn_hidden_act = gnn_hidden_act
        self.gnn_mlp_type = gnn_mlp_type
        self.pretraining_tp = pretraining_tp




class GOFAMistralConfig(MistralConfig):
    def __init__(self, dim=4096, num_layers=6, mem_token=128, head=32, add_self_loops=True, dropout=0.0,
                 llama_dtype=torch.bfloat16, gnn_hidden_act="relu", gnn_mlp_type="gp", gnn_type="index", position_encoding="rotary", pretraining_tp=0, gating=True, interleave=True, mp_att="concat", trainable_layer=5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mem_token = mem_token
        self.head = head
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.num_layers = num_layers
        self.llama_dtype = llama_dtype
        self.gnn_hidden_act = gnn_hidden_act
        self.gnn_mlp_type = gnn_mlp_type
        self.gnn_type = gnn_type
        self.pretraining_tp = pretraining_tp
        self.position_encoding = position_encoding
        self.interleave = interleave
        self.gating = gating
        self.mp_att = mp_att
        self.trainable_layer = trainable_layer
