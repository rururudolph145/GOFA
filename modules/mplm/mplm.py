import torch
from torch import nn as nn
from transformers import BitsAndBytesConfig

from modules.mplm.mplm_modeling import MPLMSparseForCausalLM


class MPLM(torch.nn.Module):
    def __init__(self, transformer_args):
        super().__init__()
        model_args, training_args, gofa_args = transformer_args
        model = MPLMSparseLora(model_args, training_args, gofa_args)  # restored llama2-7b-chat model

        self.model = model
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.left_tokenizer.pad_token = self.model.left_tokenizer.bos_token

    def get_tokenizer(self):
        return self.model.tokenizer

    def merge_lora(self):
        self.model.merge_lora()

    def train_mode(self):
        # for param in self.model.dec.parameters():
        #     param.requires_grad = False
        pass

    def forward(self, data, masked_data=None, graph=None):
        # print(self.model.training_args.model_max_length)
        if masked_data == None:
            masked_data = data
        cur_device = self.model.icae.get_base_model().model.embed_tokens.weight.device
        prompt_output_full = self.model.tokenizer(data, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        prompt_output_masked = self.model.tokenizer(masked_data, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        prompt_output_masked = [p + [self.model.tokenizer.eos_token_id] if len(p) == len(prompt_output_full[i]) else p for i, p in enumerate(prompt_output_masked)]
        prompt_output_full = [p + [self.model.tokenizer.eos_token_id] for p in prompt_output_full]
        prompt_output_ids_full = []
        prompt_mask = []
        prompt_masked_mask = []
        for i, p in enumerate(prompt_output_full):
            if len(p) < self.model.training_args.model_max_length:
                prompt_output_id = p
                prompt_masked_mask_loc = len(prompt_output_masked[i])
            elif self.model.training_args.model_max_length/2 < len(prompt_output_masked[i]) != len(p):
                extra = len(p) - self.model.training_args.model_max_length
                offset = min(extra, int(len(prompt_output_masked[i])-self.model.training_args.model_max_length/2))
                prompt_output_id = p[offset: offset + self.model.training_args.model_max_length]
                prompt_masked_mask_loc = len(prompt_output_masked[i]) - offset
            else:
                prompt_output_id = p[:self.model.training_args.model_max_length]
                prompt_masked_mask_loc = min(len(prompt_output_masked[i]), self.model.training_args.model_max_length)
            prompt_output_ids_full.append(prompt_output_id)
            prompt_mask.append([True] * len(prompt_output_id))
            prompt_masked_mask.append(([True] * prompt_masked_mask_loc + [False] * (len(prompt_output_id) - prompt_masked_mask_loc)))


        # print(self.model.tokenizer.batch_decode(prompt_output_ids_full))

        prompt_output = self.model.tokenizer.pad({"input_ids":prompt_output_ids_full, "attention_mask": prompt_mask}, padding=True, return_tensors="pt")

        output_masked = self.model.tokenizer.pad({"input_ids":prompt_output_ids_full, "attention_mask": prompt_masked_mask}, padding=True, return_tensors="pt")

        token_count = output_masked["attention_mask"].sum(dim=-1).to(graph.edge_map.device)

        node_token_count = token_count[:graph.num_node_feat]
        edge_token_count = token_count[graph.num_node_feat:][graph.edge_map]

        max_edge_token_count = edge_token_count.max()

        edge_order = torch.zeros(len(graph.edge_index[0]), device=graph.edge_index.device, dtype=torch.long)

        node_order = []

        for i in range(len(graph.node_map)):
            s_target = (graph.edge_index[1] == i)
            s_node = graph.edge_index[0][s_target]
            nc = node_token_count[s_node]
            ec = edge_token_count[s_target]
            tc = nc + ec
            tc = torch.cat([torch.tensor([0], device=tc.device), tc[:-1]])
            tc = torch.cumsum(tc, dim=0)
            edge_order[s_target] = tc
            node_order.append(tc[-1:])
        graph.edge_order = edge_order
        graph.node_order = torch.cat(node_order)
        graph.max_edge_token = max_edge_token_count
        att_mask = output_masked["attention_mask"].to(cur_device)
        mem_mask = att_mask[:graph.num_node_feat]
        edge_mask = att_mask[graph.num_node_feat:][:, :graph.max_edge_token]

        prompt_answer_ids = prompt_output["input_ids"].to(cur_device)

        prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(prompt_answer_ids)

        output_emb = self.model.icae(inputs_embeds=prompt_answer_embs, graph=graph, partial_grad=True, map_node=True, mem_mask=mem_mask, edge_mask=edge_mask, use_cache=False).logits

        final_mask = torch.logical_xor(prompt_output["attention_mask"], output_masked["attention_mask"])
        # print(self.model.tokenizer.decode(prompt_output["input_ids"].to(cur_device)[final_mask.to(torch.bool)]))
        return output_emb[:, :-1], prompt_output["input_ids"][:, 1:].to(cur_device), final_mask[:, 1:].to(torch.bool)

    def encode(self, data, input, prompt=None):
        raise NotImplementedError("no encdoe for llama")


    def decode(self, data, input, prompt=None):
        return self(data, input, prompt)

    def generate(self, data, graph=None, target_index=None):
        target_index = target_index.to("cpu")
        target_set = set(target_index.numpy())
        cur_device = self.model.icae.get_base_model().model.embed_tokens.weight.device
        prompt_output_full = self.model.tokenizer(data, add_special_tokens=False, padding=False, truncation=False)[
            "input_ids"]
        prompt_output_ids_full = [
            p + [self.model.tokenizer.eos_token_id] if (len(p) < self.model.training_args.model_max_length and i not in target_set) else p[
                                                                                                               :self.model.training_args.model_max_length]
            for i, p in enumerate(prompt_output_full)]
        prompt_mask = [[True] * len(p) for p in prompt_output_ids_full]

        prompt_output = self.model.tokenizer.pad({"input_ids": prompt_output_ids_full, "attention_mask": prompt_mask},
                                                 padding=True, return_tensors="pt")

        token_count = prompt_output["attention_mask"].sum(dim=-1).to(graph.edge_map.device)

        node_token_count = token_count[:graph.num_node_feat]
        edge_token_count = token_count[graph.num_node_feat:][graph.edge_map]

        max_edge_token_count = edge_token_count.max()

        edge_order = torch.zeros(len(graph.edge_index[0]), device=graph.edge_index.device, dtype=torch.long)

        node_order = []

        for i in range(len(graph.node_map)):
            s_target = (graph.edge_index[1] == i)
            s_node = graph.edge_index[0][s_target]
            nc = node_token_count[s_node]
            ec = edge_token_count[s_target]
            tc = nc + ec
            tc = torch.cat([torch.tensor([0], device=tc.device), tc[:-1]])
            tc = torch.cumsum(tc, dim=0)
            edge_order[s_target] = tc
            node_order.append(tc[-1:])
        graph.edge_order = edge_order
        graph.node_order = torch.cat(node_order)
        graph.max_edge_token = max_edge_token_count

        prompt_answer_ids = prompt_output["input_ids"].to(cur_device)

        # print(self.model.tokenizer.batch_decode(torch.cat([prompt_output["input_ids"][:graph.num_node_feat][graph.node_map.cpu().numpy()], prompt_output["input_ids"][graph.num_node_feat:]])))

        prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(prompt_answer_ids)

        past_key_values = None

        eos_reached = torch.zeros(len(target_index), dtype=torch.bool).to(prompt_answer_embs.device)

        att_mask = prompt_output["attention_mask"].to(cur_device)

        generate_text = []
        mem_mask = att_mask[:graph.num_node_feat]
        edge_mask = att_mask[graph.num_node_feat:][:, :graph.max_edge_token]
        for i in range(128):
            if i == 0:
                out = self.model.icae(inputs_embeds=prompt_answer_embs, graph=graph, partial_grad=True, attention_mask=att_mask,
                                             map_node=True, mem_mask=mem_mask, use_cache=True, edge_mask=edge_mask)
                logits = out.logits[:graph.num_node_feat, :, :32000][torch.arange(graph.num_node_feat), node_token_count-1]
                past_key_values = out.past_key_values
                past_key_values.remove_edge(graph.num_node_feat)
                att_mask = att_mask[:graph.num_node_feat][graph.node_map.cpu()]
            else:
                position_ids = att_mask.clone().to(torch.long)
                position_ids[:, 0] = 0
                position_ids = torch.cumsum(position_ids, dim=-1)[:, -1:]
                out = self.model.icae(inputs_embeds=prompt_answer_embs, graph=graph, use_cache=True, past_key_values=past_key_values, attention_mask=att_mask, map_node=True, mem_mask=mem_mask, edge_mask=edge_mask, partial_grad=True, position_ids=position_ids)
                logits = out.logits[:, -1, :32000]
                past_key_values = out.past_key_values

            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            eos_reached = torch.logical_or(eos_reached, (next_token_id[target_index] == self.model.tokenizer.eos_token_id).view(-1))

            prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(next_token_id).to(prompt_answer_embs.device)

            generate_text.append(next_token_id.view(-1, 1))

            add_att_mask = torch.zeros((len(att_mask), 1), dtype=att_mask.dtype, device=att_mask.device)
            add_att_mask[target_index] = 1
            att_mask = torch.cat([att_mask, add_att_mask], dim=-1)
            mem_mask = torch.cat([mem_mask, add_att_mask], dim=-1)

            if torch.all(eos_reached):
                break

        generate_text = torch.cat(generate_text, dim=-1)
        generate_text[generate_text >= 32000] = 1

        generated_text = self.model.tokenizer.batch_decode(generate_text)
        generated_text = [generated_text[i] for i in target_index]

        return generated_text

    def extract_content_after_inst(self, generated_text):
        # Find the index of the closing tag [/INST]
        closing_tag = "[/INST]"
        start_index = generated_text.find(closing_tag)

        if start_index == -1:
            # If the closing tag is not found, return the entire text
            return generated_text

        # Extract the content after the closing tag
        content_after_inst = generated_text[start_index + len(closing_tag):].strip()

        return content_after_inst


class MPLMSparseLora(nn.Module):
    def __init__(self, model_args, training_args, gofa_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        # self.auto_encoder = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.quantization = model_args.quantization
        self.icae = MPLMSparseForCausalLM.from_pretrained(self.model_name, gofa_config,
                                                           torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16,
                                                           use_flash_attention_2=False, resume_download=False)
        self.icae.model.align_weight()
        self.icae.lm_head.requires_grad_(False)
        self.icae.model.embed_tokens.requires_grad_(False)
        self.eos_id = 1
        self.dim = self.icae.config.hidden_size
        self.prepare_lora()
        # if self.quantization:
        #     self.icae = prepare_model_for_kbit_training(self.icae)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.left_tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.left_tokenizer.padding_side = "left"
        self.left_tokenizer.truncation_side = "left"

    def prepare_lora(self):
        if self.model_args.dec_lora:
            lora_config = self.create_lora_config()
            self.icae = get_peft_model(self.icae, lora_config)
            self.icae.get_base_model().model.set_trainable_state(lora=True)
        else:
            self.icae.get_base_model().model.set_trainable_state(lora=False)
        print(self.icae.get_base_model().model.g_layers.state_dict()['0.0.self_attn.gq_proj.base_layer.weight'])

    def merge_lora(self):
        self.icae = self.icae.merge_and_unload()
        self.prepare_lora()

    def create_bnb_config(self):
        """
        quantization configuration.
        """
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16)
        # bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        return bnb_config

    def create_lora_config(self):
        lora_config = LoraConfig(

            r=128,

            lora_alpha=32,

            lora_dropout=self.model_args.lora_dropout,

            bias="none",

            task_type="CAUSAL_LM",

            target_modules=["gq_proj", "gk_proj", "gv_proj"]

        )
        return lora_config

    def get_tokenizer(self):
        return self.tokenizer
