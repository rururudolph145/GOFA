import torch

from modules.llm.mistral_lora import MistralLora


class MistralHelper(torch.nn.Module):
    def __init__(self, transformer_args):
        super().__init__()
        model_args, training_args, gofa_args = transformer_args
        model = MistralLora(model_args, training_args, gofa_args)  # restored llama2-7b-chat model

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
