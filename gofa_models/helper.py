# example code for running inference with fine-tuned checkpoint
import numpy as np
import torch
from modules.gofa_icae_llama_modeling import LlamaICAE
from modules.gofa_icae_mistral_modeling import MistralICAE
from modules.llama_modeling import LlamaLora, MPLMLora, MPLMSparseLora
from collections import OrderedDict
from safetensors.torch import load_file


class GOFALlamaHelper(torch.nn.Module):
    def __init__(self, transformer_args):
        super().__init__()
        model_args, training_args, gofa_args = transformer_args
        model = LlamaICAE(model_args, training_args, gofa_args)  # restored llama2-7b-chat model
        state_dict = torch.load(model_args.llama_pretrain_checkpoint)  # change the path for your model
        new_state_dict = OrderedDict()

        for layer_name, weight in state_dict.items():
            if isinstance(weight, torch.Tensor) or weight != 0.0:
                new_state_dict[layer_name.replace("default", "encadapt")] = weight
        model.load_state_dict(new_state_dict, strict=False)
        # model.merge_lora()
        self.dec_lora = model_args.dec_lora
        self.mem_tokens = list(range(model.vocab_size, model.vocab_size + model_args.mem_size))
        self.mem_size = model_args.mem_size
        self.model = model
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.left_tokenizer.pad_token = self.model.left_tokenizer.bos_token
        for param in self.model.icae.parameters():
            param.requires_grad = False
        for param in self.model.icae.get_base_model().model.g_layers.parameters():
            param.requires_grad = True
        if self.dec_lora:
            for name, param in self.model.icae.named_parameters():
                if "default" in name:
                    param.requires_grad = True

    def get_tokenizer(self):
        return self.model.tokenizer

    def train_mode(self):
        # for param in self.model.dec.parameters():
        #     param.requires_grad = False
        self.model.icae.set_adapter("encadapt")
        for param in self.model.icae.parameters():
            param.requires_grad = False

    def forward(self, data, answer, edge_data, prompt=None, graph=None, partial_grad=None):
        cur_device = self.model.memory_token_embed.weight.device
        batch_size = len(data)
        if prompt is None:
            prompt = [""] * len(data)

        text_input = self.model.tokenizer(data, truncation=True, max_length=self.model.training_args.model_max_length,
                                          padding=False, return_attention_mask=False)["input_ids"]
        text_target = \
            self.model.tokenizer(answer, truncation=True, max_length=self.model.training_args.model_max_length,
                                 padding=False, return_attention_mask=False)["input_ids"]
        edge_input = \
            self.model.tokenizer(edge_data, truncation=True, max_length=self.model.training_args.model_max_length,
                                 padding=False, return_attention_mask=False)["input_ids"] if len(edge_data) > 0 else []

        text_target = [p + [self.model.tokenizer.eos_token_id] for p in text_target]
        target_ids = torch.cat([torch.tensor(p, dtype=torch.long) for p in text_target], dim=-1).to(cur_device)

        prompt_input = self.model.tokenizer(prompt, add_special_tokens=False, padding=False)["input_ids"]
        prompt_input = [[self.model.ft_token_id] + a + [self.model.ft_token_id] if len(a) > 0 else a for a in
                        prompt_input]

        text_ids = [a + self.mem_tokens + b + c for a, b, c in zip(text_input, prompt_input, text_target)]
        target_mask = [[False] * (len(a) + self.mem_size + len(b) - 1) + [True] * (len(c)) + [False] for a, b, c in
                       zip(text_input, prompt_input, text_target)]

        edge_text_ids = [a + self.mem_tokens for a in edge_input]

        graph.num_node_feat = len(text_ids)

        input_ids = text_ids + edge_text_ids
        target_mask = target_mask + [[False] * len(a) for a in edge_text_ids]

        text_output = {"input_ids": input_ids, "attention_mask": target_mask}
        text_output = self.model.tokenizer.pad(text_output, padding=True, return_tensors="pt")
        input_ids = text_output["input_ids"].to(device=cur_device)
        target_mask = text_output["attention_mask"].to(torch.bool)
        mem_mask = torch.logical_and(input_ids >= self.model.vocab_size,
                                     input_ids < self.model.vocab_size + self.mem_size)

        mem_mask = mem_mask.to(cur_device)

        autoencoder_input_embedding = self.model.icae.get_base_model().model.embed_tokens(input_ids)
        autoencoder_input_embedding[mem_mask] = self.model.memory_token_embed(
            input_ids[mem_mask] - self.model.vocab_size).to(autoencoder_input_embedding)
        self.model.icae.enable_adapter_layers()
        compress_outputs = self.model.icae(inputs_embeds=autoencoder_input_embedding, output_hidden_states=True,
                                           graph=graph, mem_mask=mem_mask, partial_grad=partial_grad, map_node=False)
        self.model.icae.disable_adapter_layers()

        compress_outputs = compress_outputs.logits
        return compress_outputs, target_ids, target_mask

    def encode(self, data, graph=None, partial_grad=None):
        batch_size = len(data)
        text_output = \
        self.model.tokenizer(data, truncation=True, max_length=self.model.training_args.model_max_length, padding=False,
                             return_attention_mask=False)["input_ids"]
        text_output = [t + self.mem_tokens for t in text_output]
        text_output = {"input_ids": text_output}
        text_output = self.model.tokenizer.pad(text_output, padding=True, return_tensors="pt")["input_ids"].to(
            self.model.memory_token_embed.weight.device)
        mem_mask = text_output >= self.model.vocab_size

        mem_mask = mem_mask.to(self.model.memory_token_embed.weight.device)

        autoencoder_input_embedding = self.model.icae.get_base_model().model.embed_tokens(text_output)
        autoencoder_input_embedding[mem_mask] = self.model.memory_token_embed(
            text_output[mem_mask] - self.model.vocab_size).to(autoencoder_input_embedding)
        self.model.icae.set_adapter("encadapt")
        self.model.icae.enable_adapter_layers()
        for name, param in self.model.icae.named_parameters():
            if "encadapt" in name:
                param.requires_grad = False
        compress_outputs = self.model.icae(inputs_embeds=autoencoder_input_embedding, output_hidden_states=True,
                                           graph=graph, mem_mask=mem_mask, partial_grad=partial_grad, map_node=True)
        self.model.icae.disable_adapter_layers()
        compress_outputs = compress_outputs.hidden_states[-1]
        if graph is not None:
            node_emb = compress_outputs[:len(graph.node_map)]
            map_mem_mask = mem_mask[:graph.num_node_feat][graph.node_map]
            memory_embedding = node_emb[map_mem_mask].view(len(node_emb), self.mem_size, -1)
        else:
            memory_embedding = compress_outputs[mem_mask].view(batch_size, self.mem_size, -1)
        return memory_embedding

    def llm_output(self, data, input, prompt=None):
        self.model.icae.disable_adapter_layers()
        cur_device = self.model.memory_token_embed.weight.device
        prompt_output = self.model.tokenizer(data, add_special_tokens=False, padding=False, truncation=True,
                                             max_length=self.model.training_args.model_max_length)["input_ids"]
        input_tokens = self.model.tokenizer(input, add_special_tokens=False, padding=False, truncation=True,
                                            max_length=self.model.training_args.model_max_length)["input_ids"]
        prompt_output = [p + [self.model.tokenizer.eos_token_id] for p in prompt_output]
        if prompt is None:
            prompt = [""] * len(data)
        prompt_input = self.model.tokenizer(prompt, add_special_tokens=False, padding=False)["input_ids"]
        prompt_input = [[self.model.ft_token_id] + a + [self.model.ft_token_id] if len(a) > 0 else a for a in
                        prompt_input]
        prompt_ids = [a + b + c for a, b, c in zip(input_tokens, prompt_input, prompt_output)]
        prompt_mask = [[False] * (len(a) + len(b) - 1) + [True] * (len(c)) + [False] for a, b, c in
                       zip(input_tokens, prompt_input, prompt_output)]
        answer_prompt = torch.cat([torch.tensor(p, dtype=torch.long) for p in prompt_output], dim=-1).to(cur_device)

        prompt_output = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
        prompt_output = self.model.tokenizer.pad(prompt_output, padding=True, return_tensors="pt")

        prompt_answer_ids = prompt_output["input_ids"].to(cur_device)

        target_mask = prompt_output["attention_mask"].to(cur_device).to(torch.bool)

        prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(prompt_answer_ids)

        output_emb = self.model.dec(inputs_embeds=prompt_answer_embs).logits

        return output_emb, answer_prompt, target_mask

    def decode(self, data, mem_embs, graph=None, prompt=None):
        prompt_output = self.model.tokenizer(data, add_special_tokens=False, padding=False, truncation=True,
                                             max_length=self.model.training_args.model_max_length)["input_ids"]
        prompt_output = [p + [self.model.tokenizer.eos_token_id] for p in prompt_output]
        if prompt is None:
            prompt = [""] * len(data)
        prompt_input = self.model.left_tokenizer(prompt, add_special_tokens=False, padding=False, truncation=True, max_length=512)["input_ids"]
        # print(self.model.left_tokenizer.batch_decode(prompt_input))
        prompt_input = [[self.model.ft_token_id] + a + [self.model.ft_token_id] if len(a) > 0 else a for a in
                        prompt_input]
        prompt_ids = [a + b for a, b in zip(prompt_input, prompt_output)]
        prompt_mask = [[False] * len(a) + [True] * (len(b)) + [False] for a, b in zip(prompt_input, prompt_output)]
        mem_mask = torch.tensor([[False] * (self.mem_size - 1) for _ in prompt_output], dtype=torch.long).to(mem_embs.device)
        answer_prompt = torch.cat([torch.tensor(p, dtype=torch.long) for p in prompt_output], dim=-1).to(
            mem_embs.device)
        prompt_output = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
        prompt_output = self.model.tokenizer.pad(prompt_output, padding=True, return_tensors="pt")
        prompt_answer_ids = prompt_output["input_ids"].to(mem_embs.device)
        special_prompt = prompt_answer_ids >= self.model.vocab_size
        target_mask = torch.cat([mem_mask, prompt_output["attention_mask"].to(mem_mask)], dim=-1).to(torch.bool)
        prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
        prompt_answer_embs[special_prompt] = self.model.memory_token_embed(
            prompt_answer_ids[special_prompt] - self.model.vocab_size).to(prompt_answer_embs)
        decode_embed = torch.cat([mem_embs.to(prompt_answer_embs), prompt_answer_embs], dim=1)
        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        output_emb = self.model.icae(inputs_embeds=decode_embed).logits

        return output_emb, answer_prompt, target_mask

    def generate(self, mem_embs, graph=None, prompt=None):
        if prompt is None:
            prompt = [""] * len(mem_embs)
        prompt_input = self.model.tokenizer(prompt, add_special_tokens=False, padding=False)["input_ids"]
        prompt_ids = [[self.model.ft_token_id] + a + [self.model.ft_token_id] if len(a) > 0 else a for a in
                      prompt_input]

        mem_mask = [[True] * self.mem_size + [False] * len(a) for a in prompt_ids]
        att_mask = [[True] * (self.mem_size + len(a)) for a in prompt_ids]
        prompt_ids = [[self.model.tokenizer.pad_token_id] * self.mem_size + a for a in prompt_ids]
        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": mem_mask},
                                                         padding=True, return_tensors="pt")
        mem_mask = input_prompt_ids["attention_mask"].to(device=mem_embs.device, dtype=torch.bool)

        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": att_mask},
                                                         padding=True, return_tensors="pt")

        prompt_ids = input_prompt_ids["input_ids"]
        att_mask = input_prompt_ids["attention_mask"].to(device=mem_embs.device)

        prompt_answer_ids = prompt_ids.to(device=mem_embs.device, dtype=torch.long)
        special_prompt = prompt_answer_ids >= self.model.vocab_size
        prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
        prompt_answer_embs[special_prompt] = self.model.memory_token_embed(
            prompt_answer_ids[special_prompt] - self.model.vocab_size).to(prompt_answer_embs)

        prompt_answer_embs[mem_mask] = mem_embs.view(-1, mem_embs.size()[-1])

        # decode_embed = torch.cat([mem_embs.to(prompt_answer_embs), prompt_answer_embs], dim=1)
        decode_embed = prompt_answer_embs
        output = decode_embed.clone()

        generate_text = []
        eos_reached = torch.zeros(len(output), dtype=torch.bool).to(output.device)

        past_key_values = None
        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        for i in range(128):
            out = self.model.icae(inputs_embeds=output, attention_mask=att_mask, past_key_values=past_key_values,
                                 use_cache=True)

            logits = out.logits[:, -1]

            past_key_values = out.past_key_values

            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            eos_reached = torch.logical_or(eos_reached, (next_token_id == self.model.tokenizer.eos_token_id).view(-1))

            eos_reached = torch.logical_or(eos_reached, (next_token_id == self.model.tokenizer.bos_token_id).view(-1))

            eos_reached = torch.logical_or(eos_reached, (next_token_id >= 32000).view(-1))
            generate_text.append(next_token_id.view(-1, 1))
            if torch.all(eos_reached):
                break

            output = self.model.icae.get_base_model().model.embed_tokens(next_token_id).to(mem_embs.device)

            att_mask = torch.cat(
                [att_mask, torch.ones((len(att_mask), 1), dtype=att_mask.dtype, device=att_mask.device)], dim=-1)
        generate_text = torch.cat(generate_text, dim=-1)
        generate_text[generate_text >= 32000] = 1

        generated_text = self.model.tokenizer.batch_decode(generate_text)

        return generated_text


class GOFAMistralHelper(torch.nn.Module):
    def __init__(self, transformer_args):
        super().__init__()
        model_args, training_args, gofa_args = transformer_args
        model = MistralICAE(model_args, training_args, gofa_args)  # restored llama2-7b-chat model
        state_dict = load_file(model_args.mistral_pretrain_checkpoint)  # change the path for your model
        new_state_dict = OrderedDict()
        for layer_name, weight in state_dict.items():
            new_state_dict[layer_name.replace("default", "encadapt")] = weight
        model.load_state_dict(new_state_dict, strict=False)
        # model.merge_lora()
        self.dec_lora = model_args.dec_lora
        self.mem_tokens = list(range(model.vocab_size, model.vocab_size + model_args.mem_size))
        self.mem_size = model_args.mem_size
        self.model = model
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.left_tokenizer.pad_token = self.model.left_tokenizer.bos_token
        for param in self.model.icae.parameters():
            param.requires_grad = False
        for param in self.model.icae.get_base_model().model.g_layers.parameters():
            param.requires_grad = True
        if self.dec_lora:
            for name, param in self.model.icae.named_parameters():
                if "default" in name:
                    param.requires_grad = True

    def get_tokenizer(self):
        return self.model.tokenizer

    def train_mode(self):
        self.model.icae.set_adapter("encadapt")
        for param in self.model.icae.parameters():
            param.requires_grad = False

    def forward(self, data, answer, edge_data, prompt=None, graph=None, partial_grad=None):
        cur_device = self.model.memory_token_embed.weight.device
        batch_size = len(data)
        if prompt is None:
            prompt = [""] * len(data)

        text_input = \
            self.model.tokenizer(data, truncation=True, max_length=5120, padding=False, return_attention_mask=False)[
                "input_ids"]
        text_target = \
            self.model.tokenizer(answer, truncation=True, max_length=self.model.training_args.model_max_length,
                                 padding=False, return_attention_mask=False)["input_ids"]
        edge_input = \
            self.model.tokenizer(edge_data, truncation=True, max_length=self.model.training_args.model_max_length,
                                 padding=False, return_attention_mask=False)["input_ids"] if len(edge_data) > 0 else []

        text_target = [p + [self.model.tokenizer.eos_token_id] for p in text_target]
        target_ids = torch.cat([torch.tensor(p, dtype=torch.long) for p in text_target], dim=-1).to(cur_device)

        prompt_input = self.model.tokenizer(prompt, add_special_tokens=False, padding=False)["input_ids"]
        prompt_left_ids = [[1, 733, 16289, 28793]]
        prompt_right_ids = [[self.model.ft_token_id] + a + [733, 28748, 16289, 28793] if len(a) > 0 else a for a in
                            prompt_input]
        prompt_right_ids = torch.LongTensor([prompt_right_ids]).to(cur_device)

        text_ids = [a + b + self.mem_tokens + c + d for a, b, c, d in
                    zip(text_input, prompt_left_ids, prompt_input, text_target)]
        print(text_ids)
        target_mask = [[False] * (len(a) + self.mem_size + len(b) + len(c) - 1) + [True] * (len(d)) + [False] for
                       a, b, c, d in zip(text_input, prompt_left_ids, prompt_input, text_target)]
        edge_text_ids = [a + self.mem_tokens for a in edge_input]

        graph.num_node_feat = len(text_ids)
        print(graph.num_node_feat)

        input_ids = text_ids + edge_text_ids
        target_mask = target_mask + [[False] * len(a) for a in edge_text_ids]

        text_output = {"input_ids": input_ids, "attention_mask": target_mask}
        text_output = self.model.tokenizer.pad(text_output, padding=True, return_tensors="pt")
        input_ids = text_output["input_ids"].to(device=cur_device)
        target_mask = text_output["attention_mask"].to(torch.bool)
        mem_mask = torch.logical_and(input_ids >= self.model.vocab_size,
                                     input_ids < self.model.vocab_size + self.mem_size)

        mem_mask = mem_mask.to(cur_device)

        autoencoder_input_embedding = self.model.icae.get_base_model().model.embed_tokens(input_ids)
        autoencoder_input_embedding[mem_mask] = self.model.memory_token_embed(
            input_ids[mem_mask] - self.model.vocab_size).to(autoencoder_input_embedding)
        self.model.icae.enable_adapter_layers()
        print(autoencoder_input_embedding.shape)
        print('---' * 30)

        compress_outputs = self.model.icae(inputs_embeds=autoencoder_input_embedding, output_hidden_states=True,
                                           graph=graph, mem_mask=mem_mask, partial_grad=partial_grad, map_node=False)
        self.model.icae.disable_adapter_layers()

        compress_outputs = compress_outputs.logits

        return compress_outputs[target_mask], target_ids

    def encode(self, data, graph=None, partial_grad=None):
        cur_device = self.model.memory_token_embed.weight.device
        batch_size = len(data)
        text_output = \
        self.model.tokenizer(data, truncation=True, max_length=self.model.training_args.model_max_length, padding=False,
                             return_attention_mask=False)["input_ids"]

        text_output = [t + self.mem_tokens for t in text_output]
        text_output = {"input_ids": text_output}
        text_output = self.model.tokenizer.pad(text_output, padding=True, return_tensors="pt")["input_ids"].to(
            cur_device)
        mem_mask = text_output >= self.model.vocab_size

        mem_mask = mem_mask.to(cur_device)
        autoencoder_input_embedding = self.model.tokens_to_embeddings(text_output)

        self.model.icae.set_adapter("encadapt")
        self.model.icae.enable_adapter_layers()
        for name, param in self.model.icae.named_parameters():
            if "encadapt" in name:
                param.requires_grad = False
        compress_outputs = self.model.icae(inputs_embeds=autoencoder_input_embedding, output_hidden_states=True,
                                           graph=graph, mem_mask=mem_mask, partial_grad=partial_grad, map_node=True)
        self.model.icae.disable_adapter_layers()
        compress_outputs = compress_outputs.hidden_states[-1]

        if graph is not None:
            node_emb = compress_outputs[:len(graph.node_map)]
            map_mem_mask = mem_mask[:graph.num_node_feat][graph.node_map]
            memory_embedding = node_emb[map_mem_mask].view(len(node_emb), self.mem_size, -1)
        else:
            memory_embedding = compress_outputs[mem_mask].view(batch_size, self.mem_size, -1)
        return memory_embedding

    def decode(self, data, mem_embs, graph=None, prompt=None):
        prompt_output = self.model.tokenizer(data, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        prompt_output = [p + [self.model.tokenizer.eos_token_id] if len(p) < self.model.training_args.model_max_length else p[:self.model.training_args.model_max_length] for p in prompt_output]
        original_prompt_output = prompt_output

        if prompt is None:
            prompt = [""] * len(data)
        prompt_input = self.model.left_tokenizer(prompt, add_special_tokens=False, padding=False, truncation=True, max_length=512)["input_ids"]
        batch_size = len(prompt_input)

        # For Mistral, decode contains: prefix, memory slots and suffix
        prompt_left_ids = [[1, 733, 16289, 28793] if len(a) > 0 else [] for a in prompt_input]
        prompt_right_ids = [[self.model.ft_token_id] + a + [733, 28748, 16289, 28793] if len(a) > 0 else a for a in
                            prompt_input]
        prompt_ids = [a + [self.model.tokenizer.pad_token_id] * self.mem_size + b + c for a, b, c in
                      zip(prompt_left_ids, prompt_right_ids, prompt_output)]
        prompt_mask = [
            [False] * (len(prompt_left_ids[i]) + self.mem_size - 1 + len(prompt_right_ids[i])) + [True] * len(
                prompt_output[i]) + [False] for i in range(batch_size)]

        answer_prompt = torch.cat([torch.tensor(p, dtype=torch.long) for p in prompt_output], dim=-1).to(
            mem_embs.device)

        prompt_output = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
        prompt_output = self.model.tokenizer.pad(prompt_output, padding=True, return_tensors="pt")
        prompt_answer_ids = prompt_output["input_ids"].to(mem_embs.device)
        prompt_answer_embs = self.model.tokens_to_embeddings(prompt_answer_ids)

        mem_mask = [[False] * len(prompt_left_ids[i]) + [True] * self.mem_size + [False] * (
                len(prompt_output["input_ids"][i]) - len(prompt_left_ids[i]) - self.mem_size) for i in
                    range(batch_size)]
        prompt_mask = [
            [False] * (len(prompt_left_ids[i]) + self.mem_size - 1 + len(prompt_right_ids[i])) + [True] * len(
                original_prompt_output[i]) + [False] * (1 + len(prompt_output["input_ids"][i]) - len(prompt_ids[i])) for
            i in range(batch_size)]

        prompt_answer_embs[torch.tensor(mem_mask)] = mem_embs.view(-1, mem_embs.size()[-1])

        target_mask = torch.tensor(prompt_mask, dtype=torch.long, device=mem_embs.device).to(torch.bool)

        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        output_emb = self.model.icae(inputs_embeds=prompt_answer_embs).logits

        return output_emb, answer_prompt, target_mask

    def generate(self, mem_embs, graph=None, prompt=None):
        cur_device = self.model.memory_token_embed.weight.device

        if prompt is None:
            prompt = [""] * len(mem_embs)
        prompt_input = self.model.tokenizer(prompt, add_special_tokens=False, padding=False)["input_ids"]
        batch_size = len(prompt_input)

        prompt_left_ids = [[1, 733, 16289, 28793] if len(a) > 0 else [] for a in prompt_input]

        prompt_right_ids = [[self.model.ft_token_id] + a + [733, 28748, 16289, 28793] if len(a) > 0 else a for a in
                            prompt_input]

        mem_mask = [[False] * len(prompt_left_ids[i]) + [True] * self.mem_size + [False] * len(prompt_right_ids[i]) for
                    i in range(batch_size)]
        att_mask = [[True] * (len(prompt_left_ids[i]) + self.mem_size + len(prompt_right_ids[i])) for i in
                    range(batch_size)]
        prompt_ids = [prompt_left_ids[i] + [self.model.tokenizer.pad_token_id] * self.mem_size + prompt_right_ids[i] for
                      i in range(batch_size)]

        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": mem_mask},
                                                         padding=True, return_tensors="pt")
        mem_mask = input_prompt_ids["attention_mask"].to(device=mem_embs.device, dtype=torch.bool)

        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": att_mask},
                                                         padding=True, return_tensors="pt")
        prompt_ids = input_prompt_ids["input_ids"]
        att_mask = input_prompt_ids["attention_mask"].to(device=mem_embs.device)

        prompt_answer_ids = prompt_ids.to(device=mem_embs.device, dtype=torch.long)
        prompt_answer_embs = self.model.tokens_to_embeddings(prompt_answer_ids)
        prompt_answer_embs[mem_mask] = mem_embs.view(-1, mem_embs.size()[-1])

        decode_embed = prompt_answer_embs
        output = decode_embed.clone()

        generate_text = []
        eos_reached = torch.zeros(len(output), dtype=torch.bool).to(output.device)

        past_key_values = None
        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        for i in range(128):
            out = self.model.icae(inputs_embeds=output, attention_mask=att_mask, past_key_values=past_key_values,
                                 use_cache=True)

            logits = out.logits[:, -1, :self.model.vocab_size - 1]

            past_key_values = out.past_key_values

            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            eos_reached = torch.logical_or(eos_reached, (next_token_id == self.model.tokenizer.eos_token_id).view(-1))

            # eos_reached = torch.logical_or(eos_reached, (next_token_id==self.model.tokenizer.bos_token_id).view(-1))

            # eos_reached = torch.logical_or(eos_reached, (next_token_id>=32000).view(-1))

            output = self.model.icae.get_base_model().model.embed_tokens(next_token_id).to(mem_embs.device)

            generate_text.append(next_token_id.view(-1, 1))
            att_mask = torch.cat(
                [att_mask, torch.ones((len(att_mask), 1), dtype=att_mask.dtype, device=att_mask.device)], dim=-1)

            if torch.all(eos_reached):
                break

        generate_text = torch.cat(generate_text, dim=-1)
        generate_text[generate_text >= 32000] = 1

        generated_text = self.model.tokenizer.batch_decode(generate_text)

        return generated_text


class LlamaHelper(torch.nn.Module):
    def __init__(self, transformer_args):
        super().__init__()
        model_args, training_args, gofa_args = transformer_args
        model = LlamaLora(model_args, training_args, gofa_args)  # restored llama2-7b-chat model

        self.model = model
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.left_tokenizer.pad_token = self.model.left_tokenizer.bos_token

    def get_tokenizer(self):
        return self.model.tokenizer

    def train_mode(self):
        # for param in self.model.dec.parameters():
        #     param.requires_grad = False
        pass

    def forward(self, data, input, prompt=None):
        # print(self.model.training_args.model_max_length)
        cur_device = self.model.icae.get_base_model().model.embed_tokens.weight.device
        prompt_output = self.model.tokenizer(data, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        input_tokens = self.model.tokenizer(input, add_special_tokens=False, padding=False, truncation=True,
                                      max_length=self.model.training_args.model_max_length)["input_ids"]
        prompt_output = [p + [self.model.tokenizer.eos_token_id] if len(p) < self.model.training_args.model_max_length else p[:self.model.training_args.model_max_length] for p in prompt_output]
        if prompt is None:
            prompt = [""] * len(data)
        prompt_input = self.model.tokenizer(prompt, add_special_tokens=False, padding=False)["input_ids"]
        prompt_ids = [a + b + c for a, b, c in zip(input_tokens, prompt_input, prompt_output)]
        prompt_mask = [[False] * (len(a) + len(b) - 1) + [True] * (len(c)) + [False] for a, b, c in
                       zip(input_tokens, prompt_input, prompt_output)]
        answer_prompt = torch.cat([torch.tensor(p, dtype=torch.long) for p in prompt_output], dim=-1).to(cur_device)

        prompt_output = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
        prompt_output = self.model.tokenizer.pad(prompt_output, padding=True, return_tensors="pt")

        prompt_answer_ids = prompt_output["input_ids"].to(cur_device)

        target_mask = prompt_output["attention_mask"].to(cur_device).to(torch.bool)

        prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(prompt_answer_ids)

        output_emb = self.model.icae(inputs_embeds=prompt_answer_embs).logits
        # for name, p in self.model.named_parameters():
        #     if "default" in name:
        #         print(p.abs().sum())
        #         break

        return output_emb, answer_prompt, target_mask

    def encode(self, data, input, prompt=None):
        raise NotImplementedError("no encdoe for llama")


    def decode(self, data, input, prompt=None):
        return self(data, input, prompt)

    def generate(self, input, prompt=None):
        cur_device = self.model.icae.get_base_model().model.embed_tokens.weight.device
        if prompt is None:
            prompt = [""] * len(input)
        prompt_ids = self.model.left_tokenizer(input, add_special_tokens=False, padding=False, truncation=True,
                                      max_length=self.model.training_args.model_max_length)["input_ids"]

        att_mask = [[True] * (len(a)) for a in prompt_ids]

        input_prompt_ids = self.model.tokenizer.pad({"input_ids": prompt_ids, "attention_mask": att_mask},
                                                         padding=True, return_tensors="pt")

        prompt_ids = input_prompt_ids["input_ids"]
        att_mask = input_prompt_ids["attention_mask"].to(device=cur_device)

        prompt_answer_ids = prompt_ids.to(device=cur_device, dtype=torch.long)
        #
        # generated_text = [self.model.tokenizer.decode(output, skip_special_tokens=True) for i, output in enumerate(prompt_ids)]
        # for s in generated_text:
        #     print("-")
        #     print(s)

        with torch.no_grad():
            outputs = self.model.icae.generate(input_ids=prompt_answer_ids, max_length=2548, num_return_sequences=1, pad_token_id = self.model.eos_id, attention_mask=att_mask)
        generated_text = [self.model.tokenizer.decode(output[len(prompt_answer_ids[i]):], skip_special_tokens=True) for i, output in enumerate(outputs)]
        # generated_text = ["---Output---".join(t.split("---Output---")[1:]) for i, t in enumerate(generated_text)]
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


def mplm_4d_causal(max_size, all_size, center_size):
    causal_mask = torch.zeros((max_size, all_size - center_size), dtype=torch.float32)
    mask = torch.full((max_size, max_size), torch.finfo(torch.float32).min)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    return torch.cat([causal_mask, mask[:, : center_size]], dim=-1)

def mplm_4d_causal_inverse(max_size, all_size, center_size):
    causal_mask = torch.zeros((max_size, all_size - center_size), dtype=torch.float32)
    mask = torch.full((max_size, max_size), torch.finfo(torch.float32).min)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    return torch.cat([causal_mask, mask[:, max_size-center_size: ]], dim=-1)

class MPLMHelper(torch.nn.Module):
    def __init__(self, transformer_args):
        super().__init__()
        model_args, training_args, gofa_args = transformer_args
        model = MPLMLora(model_args, training_args, gofa_args)  # restored llama2-7b-chat model

        self.model = model
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.left_tokenizer.pad_token = self.model.left_tokenizer.bos_token

    def get_tokenizer(self):
        return self.model.tokenizer

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

        prompt_output = self.model.tokenizer.pad({"input_ids":prompt_output_ids_full, "attention_mask": prompt_mask}, padding=True, return_tensors="pt")

        output_masked = self.model.tokenizer.pad({"input_ids":prompt_output_ids_full, "attention_mask": prompt_masked_mask}, padding=True, return_tensors="pt")

        mapped_node_feature = output_masked["attention_mask"][:graph.num_node_feat][graph.node_map.cpu()]
        mapped_edge_feature = output_masked["attention_mask"][graph.num_node_feat:]
        mapped_feature = torch.cat([mapped_node_feature, mapped_edge_feature], dim=0)

        mapped_node_feature_full = prompt_output["attention_mask"][:graph.num_node_feat][graph.node_map.cpu()]
        mapped_edge_feature_full = prompt_output["attention_mask"][graph.num_node_feat:]
        mapped_feature_full = torch.cat([mapped_node_feature_full, mapped_edge_feature_full], dim=0)

        mapped_ids = prompt_output["input_ids"][:graph.num_node_feat][graph.node_map.cpu()]
        cont_ids = torch.cat([mapped_ids, prompt_output["input_ids"][graph.num_node_feat:]], dim=0)

        max_seq_length = mapped_feature.size()[-1]

        pseudo_edge_index = torch.stack([graph.edge_index[0], graph.edge_map+len(graph.node_map), graph.edge_index[1]], dim=0).cpu()

        max_token = 0

        row_extracts = []
        col_extracts = []
        mask_extracts = []
        causal_masks = []

        for i in range(len(graph.node_map)):
            neighbor_node_and_edge = pseudo_edge_index[:2, pseudo_edge_index[2]==i].T.reshape(-1)
            mp_node_mask = mapped_feature[neighbor_node_and_edge]
            center_mask = mapped_feature_full[i:i+1]
            all_mask = torch.cat([mp_node_mask, center_mask], dim=0)
            neighbor_node_and_edge = torch.cat([neighbor_node_and_edge, torch.tensor([i])])
            individual_sizes = all_mask.sum(dim=-1)
            all_sizes = all_mask.sum()
            center_size = center_mask.sum()
            mask_extracts.append(all_sizes)
            max_token = max(max_token, all_sizes)
            row_extract = neighbor_node_and_edge.view(-1).repeat_interleave(individual_sizes)
            col_extract = all_mask.nonzero()[:, 1]
            row_extracts.append(row_extract)
            col_extracts.append(col_extract)
            causal_masks.append(mplm_4d_causal(max_seq_length, all_sizes, center_size))

        for i in range(len(graph.node_map), len(mapped_feature)):
            center_mask = mapped_feature_full[i:i + 1]
            center_size = center_mask.sum()
            row_extract = torch.zeros((center_size,), dtype=torch.long) + i
            col_extract = torch.arange(center_size)
            row_extracts.append(row_extract)
            col_extracts.append(col_extract)
            mask_extracts.append(center_size)
            causal_masks.append(mplm_4d_causal(max_seq_length, center_size, center_size))

        for i in range(len(row_extracts)):
            row_extract = torch.zeros((max_token,), dtype=torch.long)
            row_extract[:len(row_extracts[i])] = row_extracts[i]
            row_extracts[i] = row_extract
            col_extract = torch.zeros((max_token,), dtype=torch.long)
            col_extract[:len(col_extracts[i])] = col_extracts[i]
            col_extracts[i] = col_extract
            mask_extract = torch.zeros((max_token,), dtype=torch.bool)
            mask_extract[:mask_extracts[i]] = 1
            mask_extracts[i] = mask_extract
            causal_mask = torch.full((max_seq_length, max_token), torch.finfo(torch.float32).min)
            causal_mask[:, :causal_masks[i].size()[1]] = causal_masks[i]
            causal_masks[i] = causal_mask

        col_extracts = torch.stack(col_extracts)
        row_extracts = torch.stack(row_extracts)
        causal_masks = torch.stack(causal_masks)

        prompt_answer_ids = prompt_output["input_ids"].to(cur_device)

        prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(prompt_answer_ids)

        output_emb = self.model.icae(inputs_embeds=prompt_answer_embs, graph=graph, partial_grad=True, map_node=True, extracts=(row_extracts.to(cur_device), col_extracts.to(cur_device)), extract_attention_mask=causal_masks.to(cur_device), use_cache=False).logits

        final_mask = torch.logical_xor(mapped_feature, mapped_feature_full)
        return output_emb[:len(mapped_ids), :-1], mapped_ids[:, 1:].to(cur_device), final_mask[:len(mapped_ids), 1:].to(torch.bool)

    def encode(self, data, input, prompt=None):
        raise NotImplementedError("no encdoe for llama")


    def decode(self, data, input, prompt=None):
        return self(data, input, prompt)

    def generate(self, data, graph=None, target_index=None):
        # data = ["My name is jason derulo", "What is your name? Answer: ", "Question:"]
        # target_index = torch.tensor([1])
        # node_map = torch.tensor([0, 1])
        # edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        # edge_map = torch.tensor([0], dtype=torch.long)
        # graph.edge_index = edge_index.to("cuda")
        # graph.edge_map = edge_map.to("cuda")
        # graph.node_map = node_map.to("cuda")
        # graph.num_node_feat = 2
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

        prompt_output = self.model.left_tokenizer.pad({"input_ids": prompt_output_ids_full, "attention_mask": prompt_mask},
                                                 padding=True, return_tensors="pt")

        mapped_node_feature = prompt_output["attention_mask"][:graph.num_node_feat][graph.node_map.cpu()]
        mapped_edge_feature = prompt_output["attention_mask"][graph.num_node_feat:]
        mapped_feature = torch.cat([mapped_node_feature, mapped_edge_feature], dim=0)

        max_seq_length = mapped_feature.size()[-1]

        pseudo_edge_index = torch.stack([graph.edge_index[0], graph.edge_map+len(graph.node_map), graph.edge_index[1]], dim=0).cpu()

        max_token = 0

        row_extracts = []
        col_extracts = []
        mask_extracts = []
        causal_masks = []

        for i in range(len(graph.node_map)):
            neighbor_node_and_edge = pseudo_edge_index[:2, pseudo_edge_index[2] == i].T.reshape(-1)
            mp_node_mask = mapped_feature[neighbor_node_and_edge]
            center_mask = mapped_feature[i:i + 1]
            all_mask = torch.cat([mp_node_mask, center_mask], dim=0)
            neighbor_node_and_edge = torch.cat([neighbor_node_and_edge, torch.tensor([i])])
            individual_sizes = all_mask.sum(dim=-1)
            all_sizes = all_mask.sum()
            center_size = center_mask.sum()
            mask_extracts.append(all_sizes)
            max_token = max(max_token, all_sizes)
            row_extract = neighbor_node_and_edge.view(-1).repeat_interleave(individual_sizes)
            col_extract = all_mask.nonzero()[:, 1]
            row_extracts.append(row_extract)
            col_extracts.append(col_extract)
            causal_masks.append(mplm_4d_causal_inverse(max_seq_length, all_sizes, center_size))

        for i in range(len(graph.node_map), len(mapped_feature)):
            center_mask = mapped_feature[i:i + 1]
            center_size = center_mask.sum()
            max_token = max(center_size, max_token)
            row_extract = torch.zeros((center_size,), dtype=torch.long) + i
            col_extract = torch.arange(max_seq_length - center_size, max_seq_length)
            row_extracts.append(row_extract)
            col_extracts.append(col_extract)
            mask_extracts.append(center_size)
            causal_masks.append(mplm_4d_causal_inverse(max_seq_length, center_size, center_size))
        for i in range(len(row_extracts)):
            row_extract = torch.zeros((max_token,), dtype=torch.long)
            row_extract[max_token - len(row_extracts[i]):] = row_extracts[i]
            row_extracts[i] = row_extract
            col_extract = torch.zeros((max_token,), dtype=torch.long)
            col_extract[max_token - len(col_extracts[i]):] = col_extracts[i]
            col_extracts[i] = col_extract
            mask_extract = torch.zeros((max_token,), dtype=torch.bool)
            mask_extract[max_token - mask_extracts[i]:] = 1
            mask_extracts[i] = mask_extract
            causal_mask = torch.full((max_seq_length, max_token), torch.finfo(torch.float32).min)
            causal_mask[:, max_token - causal_masks[i].size()[1]:] = causal_masks[i]
            causal_masks[i] = causal_mask

        col_extracts = torch.stack(col_extracts)
        row_extracts = torch.stack(row_extracts)
        extract_att_mask = torch.stack(causal_masks)

        prompt_answer_ids = prompt_output["input_ids"].to(cur_device)

        # print(self.model.tokenizer.batch_decode(torch.cat([prompt_output["input_ids"][:graph.num_node_feat][graph.node_map.cpu().numpy()], prompt_output["input_ids"][graph.num_node_feat:]])))

        prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(prompt_answer_ids)

        past_key_values = None

        eos_reached = torch.zeros(len(target_index), dtype=torch.bool).to(prompt_answer_embs.device)

        att_mask = prompt_output["attention_mask"].to(cur_device)

        generate_text = []
        for i in range(128):
            if i == 0:
                out = self.model.icae(inputs_embeds=prompt_answer_embs, graph=graph, partial_grad=True, attention_mask=att_mask,
                                             map_node=True, extracts=(
                    row_extracts.to(cur_device), col_extracts.to(cur_device)), extract_attention_mask=extract_att_mask.to(cur_device), use_cache=True)
                logits = out.logits[target_index, -1, :32000]
                past_key_values = ()
                for i, past_val in enumerate(out.past_key_values):
                    if i < self.model.icae.get_base_model().model.n_layers:
                        past_key_values += ((past_val[0][:graph.num_node_feat][graph.node_map.cpu()][target_index], past_val[1][:graph.num_node_feat][graph.node_map.cpu()][target_index]),)
                    else:
                        past_key_values += ((past_val[0][:len(graph.node_map)][target_index], past_val[1][:len(graph.node_map)][target_index]),)
                att_mask = att_mask[:graph.num_node_feat][graph.node_map.cpu()][target_index]
                extract_att_mask = extract_att_mask[:len(graph.node_map)][target_index]
            else:
                out = self.model.icae(inputs_embeds=prompt_answer_embs, use_cache=True, past_key_values=past_key_values, attention_mask=att_mask, extract_attention_mask=extract_att_mask.to(cur_device))
                logits = out.logits[:, -1, :32000]
                past_key_values = out.past_key_values

            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            eos_reached = torch.logical_or(eos_reached, (next_token_id == self.model.tokenizer.eos_token_id).view(-1))

            prompt_answer_embs = self.model.icae.get_base_model().model.embed_tokens(next_token_id).to(prompt_answer_embs.device)

            generate_text.append(next_token_id.view(-1, 1))
            att_mask = torch.cat(
                [att_mask, torch.ones((len(att_mask), 1), dtype=att_mask.dtype, device=att_mask.device)], dim=-1)
            extract_att_mask = extract_att_mask[:,-1:]
            extract_att_mask = torch.cat([extract_att_mask, torch.full((len(extract_att_mask),1), 0).unsqueeze(-1)], dim=-1)

            if torch.all(eos_reached):
                break

        generate_text = torch.cat(generate_text, dim=-1)
        generate_text[generate_text >= 32000] = 1

        generated_text = self.model.tokenizer.batch_decode(generate_text)

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


class MPLMSparseHelper(torch.nn.Module):
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