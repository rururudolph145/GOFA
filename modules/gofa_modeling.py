from collections import OrderedDict

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import LlamaConfig
from .attn_mask import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaModel, LlamaDecoderLayer, \
    logger, BaseModelOutputWithPast, Cache, DynamicCache, _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask, CausalLMOutputWithPast, LlamaForCausalLM
from .gnn import GOFAGNNConv, GOFAGNNConvFullAtt, GOFAGNNConvCMB, GOFAGNNConvMLP, GOFADecoderLayer
from .mplm import MPLMDecoderLayer, MPLMSparseDecoderLayer, GOFACache
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralRMSNorm, MistralModel, MistralDecoderLayer, MistralForCausalLM

class GOFALlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, gofa_config):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        if gofa_config.gnn_type == "index":
            self.g_layers = nn.ModuleList([GOFAGNNConv(gofa_config) for _ in range(gofa_config.num_layers)])
        elif gofa_config.gnn_type == "full":
            self.g_layers = nn.ModuleList([GOFAGNNConvFullAtt(gofa_config) for _ in range(gofa_config.num_layers)])
        else:
            raise ValueError("Unknown GNN type for GOFA.")
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.mem_token = gofa_config.mem_token
        self.llama_dtype = gofa_config.llama_dtype

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
            raise ValueError("You cannot specify input_ids for GOFA please construct input embeddings manually")

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
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
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
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
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

        if not past_key_state and graph is not None and hidden_states.size()[1]<self.mem_token:
            raise ValueError("Running GOFA requires at least mem_token inputs.")

        cur_node_size = graph.num_node_feat if graph is not None else 0
        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            g_layer_idx = i - (len(self.layers) - len(self.g_layers))
            if g_layer_idx >= 0 and not past_key_state and graph is not None:
                if g_layer_idx == 0 and map_node:
                    hidden_states = torch.cat(
                        [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]],
                        dim=0)
                    mem_mask = torch.cat(
                        [mem_mask[:cur_node_size][graph.node_map], mem_mask[cur_node_size:]],
                        dim=0)
                    attention_mask = torch.cat([attention_mask[:cur_node_size][graph.node_map], attention_mask[cur_node_size:]], dim=0)
                    cur_node_size = len(graph.node_map)
                mem_repr = hidden_states[mem_mask].view(hidden_states.size()[0], self.mem_token, -1)
                gnn_input = mem_repr[:cur_node_size]
                gnn_edge_input = mem_repr[cur_node_size:][graph.edge_map]

                output = self.g_layers[g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input)
                output = torch.cat([output, mem_repr[cur_node_size:]], dim=0)
                gnn_output = torch.zeros_like(hidden_states, dtype=output.dtype)
                gnn_output[mem_mask] = output.view(-1, output.size()[-1])
                hidden_states = hidden_states * torch.logical_not(mem_mask).unsqueeze(2) + gnn_output
            if g_layer_idx < 0 and partial_grad:
                with torch.no_grad():
                    hidden_states = hidden_states.to(self.llama_dtype)
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            decoder_layer.__call__,
                            hidden_states,
                            attention_mask,
                            position_ids,
                            past_key_values,
                            output_attentions,
                            use_cache,
                        )
                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )

                    hidden_states = layer_outputs[0]

                    if use_cache:
                        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)
            else:
                hidden_states = hidden_states.to(self.llama_dtype)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states,
                        attention_mask, position_ids, past_key_values, output_attentions, use_cache, )
                else:
                    layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_value=past_key_values, output_attentions=output_attentions,
                        use_cache=use_cache, )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GOFALlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _keep_in_fp32_modules = ["g_layers"]

    def __init__(self, config, gofa_config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GOFALlamaModel(config, gofa_config)
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
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
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


class GOFAMistralModel(MistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: MistralConfig, gofa_config):
        super(MistralModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        if gofa_config.gnn_type == "index":
            self.g_layers = nn.ModuleList([GOFAGNNConv(gofa_config) for _ in range(gofa_config.num_layers)])
        elif gofa_config.gnn_type == "full":
            self.g_layers = nn.ModuleList([GOFAGNNConvFullAtt(gofa_config) for _ in range(gofa_config.num_layers)])
        elif gofa_config.gnn_type == "combine":
            self.g_layers = nn.ModuleList([GOFAGNNConvCMB(gofa_config) for _ in range(gofa_config.num_layers)])
        elif gofa_config.gnn_type == "mlp":
            self.g_layers = nn.ModuleList([GOFAGNNConvMLP(gofa_config) for _ in range(gofa_config.num_layers)])
        else:
            raise ValueError("Unknown GNN type for GOFA.")
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.mem_token = gofa_config.mem_token
        self.llama_dtype = gofa_config.llama_dtype
        self.n_layers = gofa_config.num_layers
        self.interleave = gofa_config.interleave
        if not self.interleave:
            self.g_layers.append(MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps))

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
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

        if not past_key_state and graph is not None and hidden_states.size()[1]<self.mem_token:
            raise ValueError("Running GOFA requires at least mem_token inputs.")

        cur_node_size = graph.num_node_feat if graph is not None else 0
        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.interleave:
                g_layer_idx = i - (len(self.layers) - self.n_layers)
                if g_layer_idx >= 0 and not past_key_state and graph is not None:
                    if g_layer_idx == 0 and map_node:
                        hidden_states = torch.cat(
                            [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]],
                            dim=0)
                        mem_mask = torch.cat(
                            [mem_mask[:cur_node_size][graph.node_map], mem_mask[cur_node_size:]],
                            dim=0)
                        attention_mask = torch.cat([attention_mask[:cur_node_size][graph.node_map], attention_mask[cur_node_size:]], dim=0)
                        cur_node_size = len(graph.node_map)
                    mem_repr = hidden_states[mem_mask].view(hidden_states.size()[0], self.mem_token, -1)
                    gnn_input = mem_repr[:cur_node_size]
                    gnn_edge_input = mem_repr[cur_node_size:][graph.edge_map]

                    output = self.g_layers[g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input)
                    output = torch.cat([output, mem_repr[cur_node_size:]], dim=0)
                    gnn_output = torch.zeros_like(hidden_states, dtype=output.dtype)
                    gnn_output[mem_mask] = output.view(-1, output.size()[-1])
                    hidden_states = hidden_states * torch.logical_not(mem_mask).unsqueeze(2) + gnn_output
            else:
                g_layer_idx = -1
            if g_layer_idx < 0 and partial_grad:
                with torch.no_grad():
                    hidden_states = hidden_states.to(self.llama_dtype)
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            decoder_layer.__call__,
                            hidden_states,
                            attention_mask,
                            position_ids,
                            past_key_values,
                            output_attentions,
                            use_cache,
                        )
                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )

                    hidden_states = layer_outputs[0]

                    if use_cache:
                        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)
            else:
                hidden_states = hidden_states.to(self.llama_dtype)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states,
                        attention_mask, position_ids, past_key_values, output_attentions, use_cache, )
                else:
                    layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_value=past_key_values, output_attentions=output_attentions,
                        use_cache=use_cache, )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)


        if not self.interleave and graph is not None:
            for g_layer_idx in range(self.n_layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                if g_layer_idx == 0 and map_node:
                    hidden_states = torch.cat(
                        [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]], dim=0)
                    mem_mask = torch.cat([mem_mask[:cur_node_size][graph.node_map], mem_mask[cur_node_size:]], dim=0)
                    attention_mask = torch.cat(
                        [attention_mask[:cur_node_size][graph.node_map], attention_mask[cur_node_size:]], dim=0)
                    cur_node_size = len(graph.node_map)
                mem_repr = hidden_states[mem_mask].view(hidden_states.size()[0], self.mem_token, -1)
                gnn_input = mem_repr[:cur_node_size]
                gnn_edge_input = mem_repr[cur_node_size:][graph.edge_map]

                output = self.g_layers[g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input)
                output = torch.cat([output, mem_repr[cur_node_size:]], dim=0)
                gnn_output = torch.zeros_like(hidden_states, dtype=output.dtype)
                gnn_output[mem_mask] = output.view(-1, output.size()[-1])
                hidden_states = hidden_states * torch.logical_not(mem_mask).unsqueeze(2) + gnn_output
            hidden_states = self.g_layers[-1](hidden_states)
        else:
            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LLMGraphCombiner(torch.nn.Module):
    def __init__(self, init_theta=0.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([init_theta]))

    def forward(self, target_feat, additional_feat, val_mask=None):
        alpha = self.theta.sigmoid()
        if val_mask is None:
            return target_feat * alpha + additional_feat*(1-alpha)
        # print(alpha)
        # print((target_feat[val_mask]**2).sum(dim=-1).mean())
        # print((additional_feat ** 2).sum(dim=-1).mean())
        output = torch.zeros_like(target_feat, dtype=additional_feat.dtype)
        output[val_mask] = additional_feat.view(-1, additional_feat.size()[-1]) * (1-alpha)

        val_multiplier = torch.zeros_like(target_feat)
        val_multiplier[torch.logical_not(val_mask)] = 1
        val_multiplier[val_mask] = alpha

        return val_multiplier * target_feat + output

class GOFAMistralCPModel(MistralModel):
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
        self.g_layers.append(nn.ModuleList([GOFADecoderLayer(gofa_config, i) for i in range(gofa_config.num_layers)]))
        self.g_layers.append(nn.ModuleList([LLMGraphCombiner() for _ in range(gofa_config.num_layers)]))
        self.g_layers.append(nn.ModuleList([MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(gofa_config.num_layers)]))
        self.g_layers.append(MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.mem_token = gofa_config.mem_token
        self.llama_dtype = gofa_config.llama_dtype
        self.n_layers = gofa_config.num_layers

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
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
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

        if not past_key_state and graph is not None and hidden_states.size()[1]<self.mem_token:
            raise ValueError("Running GOFA requires at least mem_token inputs.")

        cur_node_size = graph.num_node_feat if graph is not None else 0
        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            g_layer_idx = i - (len(self.layers) - self.n_layers)
            if g_layer_idx == 0 and map_node:
                hidden_states = torch.cat(
                    [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]], dim=0)
                mem_mask = torch.cat([mem_mask[:cur_node_size][graph.node_map], mem_mask[cur_node_size:]], dim=0)
                attention_mask = torch.cat(
                    [attention_mask[:cur_node_size][graph.node_map], attention_mask[cur_node_size:]], dim=0)
                cur_node_size = len(graph.node_map)
            if g_layer_idx >= 0 and not past_key_state and graph is not None:
                mem_repr = hidden_states[mem_mask].view(hidden_states.size()[0], self.mem_token, -1)
                gnn_input = mem_repr[:cur_node_size]
                gnn_edge_input = mem_repr[cur_node_size:][graph.edge_map]

                output = self.g_layers[0][g_layer_idx](gnn_input, graph.edge_index, gnn_edge_input)
                output = self.g_layers[2][g_layer_idx](output)
                graph_output = torch.cat([output, mem_repr[cur_node_size:]], dim=0)
                # gnn_output = torch.zeros_like(hidden_states, dtype=output.dtype)
                # gnn_output[mem_mask] = output.view(-1, output.size()[-1])
                # hidden_states = hidden_states * torch.logical_not(mem_mask).unsqueeze(2) + gnn_output
            else:
                graph_output = None
            if g_layer_idx < 0 and partial_grad:
                with torch.no_grad():
                    layer_outputs = self.forward_llm_layer(hidden_states, self.llama_dtype, decoder_layer, attention_mask, position_ids, past_key_values, output_attentions, use_cache)
            else:
                layer_outputs = self.forward_llm_layer(hidden_states, self.llama_dtype, decoder_layer, attention_mask, position_ids,
                                       past_key_values, output_attentions, use_cache)

            hidden_states = layer_outputs[0]
            if graph_output is not None:
                hidden_states = self.g_layers[1][g_layer_idx](hidden_states, graph_output, mem_mask)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        if graph is not None:
            hidden_states = self.g_layers[3](hidden_states)
        else:
            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
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


class GOFAMistralForCausalLM(MistralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _keep_in_fp32_modules = ["g_layers"]

    def __init__(self, config, gofa_config):
        super(MistralForCausalLM, self).__init__(config)
        if gofa_config.gnn_type == "cp":
            self.model = GOFAMistralCPModel(config, gofa_config)
        else:
            self.model = GOFAMistralModel(config, gofa_config)
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



def _prepare_4d_causal_attention_mask_cross(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = attention_mask.size()[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


class MPLMMistralModel(MistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: MistralConfig, gofa_config):
        super().__init__(config)
        self.g_layers = nn.ModuleList()
        self.g_layers.append(nn.ModuleList([MPLMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers-gofa_config.num_layers, config.num_hidden_layers)]))
        self.g_layers.append(MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.gradient_checkpointing = False
        self.mem_token = gofa_config.mem_token
        self.llama_dtype = gofa_config.llama_dtype
        self.n_layers = gofa_config.num_layers
        self.interleave = gofa_config.interleave
        self.trainable_layer = gofa_config.trainable_layer

        # Initialize weights and apply final processing
        self.post_init()

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
                partial_state_dict[".".join(name_split)] = source_dict[layer_name]

        self.g_layers[0].load_state_dict(partial_state_dict)
        self.g_layers[-1].load_state_dict(self.norm.state_dict())
        self.layers = self.layers[:len(self.layers) - len(self.g_layers[0])]
        self.layers.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
            partial_grad=None,
            map_node=None,
            extracts=None,
            collects=None,
            extract_attention_mask=None,
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
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
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
        if extracts is not None:
            extract_batch_size, extract_seq_length = extracts[0].size()
        else:
            extract_seq_length = seq_length

        extract_past_key_values_length = 0
        if use_cache:
            extract_past_key_values_length = past_key_values.get_usable_length(seq_length, layer_idx=len(self.layers)+len(self.g_layers[0])-1)

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        extract_position_ids = torch.arange(
            extract_past_key_values_length, extract_seq_length + extract_past_key_values_length, dtype=torch.long, device=device
        )
        extract_position_ids = extract_position_ids.unsqueeze(0)
        extract_attention_mask = extract_attention_mask.unsqueeze(1)
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
            if g_layer_idx >= 0:
                decoder_layer = self.g_layers[0][g_layer_idx]
            else:
                decoder_layer = self.layers[i]
            if g_layer_idx == 0:
                if map_node:
                    hidden_states = torch.cat(
                        [hidden_states[:cur_node_size][graph.node_map], hidden_states[cur_node_size:]], dim=0)
                    cur_node_size = len(graph.node_map)
                attention_mask = extract_attention_mask
                position_ids = extract_position_ids
            if i < len(self.layers):
                with torch.no_grad():
                    layer_outputs = self.forward_layer(g_layer_idx, map_node, hidden_states, cur_node_size, graph, past_key_state, extracts, decoder_layer, past_key_values, output_attentions, use_cache, attention_mask, position_ids)
            else:
                layer_outputs = self.forward_layer(g_layer_idx, map_node, hidden_states, cur_node_size, graph, past_key_state,
                                                   extracts, decoder_layer, past_key_values, output_attentions,
                                                   use_cache, attention_mask, position_ids)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.g_layers[-1](hidden_states)
        # hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


    def forward_layer(self, g_layer_idx, map_node, hidden_states, cur_node_size, graph, past_key_state, extracts, decoder_layer, past_key_values, output_attentions, use_cache, attention_mask, position_ids):
        if g_layer_idx >= 0 and not past_key_state and extracts is not None:
            kv_states = hidden_states[extracts[0], extracts[1]]
        else:
            kv_states = None
        hidden_states = hidden_states.to(self.llama_dtype)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask,
                                                              position_ids, past_key_values, output_attentions,
                                                              use_cache, )
        elif g_layer_idx < 0:
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                          past_key_value=past_key_values, output_attentions=output_attentions,
                                          use_cache=use_cache, )
        elif g_layer_idx >= 0:
            layer_outputs = decoder_layer(hidden_states, kv_states=kv_states, attention_mask=attention_mask,
                                          position_ids=position_ids, past_key_value=past_key_values,
                                          output_attentions=output_attentions, use_cache=use_cache, )
        return layer_outputs



class MPLMForCausalLM(MistralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _keep_in_fp32_modules = ["g_layers"]

    def __init__(self, config, gofa_config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = MPLMMistralModel(config, gofa_config)
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
        partial_grad = None,
        map_node = None, extracts=None, collects=None, extract_attention_mask=None,
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
            extracts=extracts,
            collects=collects,
            extract_attention_mask=extract_attention_mask,
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

class MPLMSparseAdaModel(MistralModel):
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
        self.g_layers.append(nn.ModuleList([MPLMSparseDecoderLayer(config, i+len(self.layers) - gofa_config.num_layers) for i in range(gofa_config.num_layers)]))
        self.g_layers.append(MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
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
        self.g_layers[-1].load_state_dict(self.norm.state_dict())

    def set_trainable_state(self, lora=False):
        for i in range(len(self.g_layers[0]) - self.trainable_layer):
            self.g_layers[0][i].requires_grad_(False)
        for i in range(len(self.layers) - self.trainable_layer):
            self.layers[i].requires_grad_(False)
        if lora:
            for i in range(len(self.g_layers[0]) - self.trainable_layer, len(self.g_layers[0])):
                for name, submodule in self.g_layers[0][i].named_modules():
                    if isinstance(submodule, MistralRMSNorm):
                        for param in submodule.parameters():
                            param.requires_grad = True
            for i in range(len(self.layers) - self.trainable_layer, len(self.layers)):
                for name, submodule in self.layers[i].named_modules():
                    if isinstance(submodule, MistralRMSNorm):
                        for param in submodule.parameters():
                            param.requires_grad = True
            self.g_layers[1].requires_grad_(True)

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
        for i in range(len(self.layers)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            g_layer_idx = i - (len(self.layers) - len(self.g_layers[0]))
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
                    layer_outputs = self.g_layers[0][g_layer_idx].forward_llm(hidden_states,
                                                                              attention_mask=attention_mask,
                                                                              position_ids=position_ids,
                                                                              output_attentions=output_attentions,
                                                                              use_cache=use_cache, )
            elif len(hidden_states) > cur_node_size:
                layer_outputs = self.g_layers[0][g_layer_idx].forward_llm(hidden_states,
                                                                          attention_mask=attention_mask,
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
                node_hs = 0.5 * graph_output + 0.5 * hidden_states[:cur_node_size]
                hidden_states = torch.cat([node_hs, hidden_states[cur_node_size:]], dim=0)
            elif hidden_states is None:
                hidden_states = graph_output

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        if graph is not None:
            hidden_states = self.g_layers[-1](hidden_states)
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