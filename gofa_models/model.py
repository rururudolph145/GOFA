import random
import re
from collections import namedtuple, OrderedDict

import torch
import numpy as np
from copy import deepcopy

from gp.nn.models.GNN import MultiLayerMessagePassing
from gp.nn.layer.pyg import RGCNEdgeConv
from gp.nn.layer.pyg import TransformerConv as MConv
from .helper import GOFALlamaHelper, GOFAMistralHelper, LlamaHelper, MPLMHelper, MPLMSparseHelper

LLM_DIM_DICT = {"ST": 768, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120, "mamba": 768, "icae": 4096,
                "icae_mem": 4096}


def add_self_loop(data ,**kwargs):
    edge_index = data.edge_index
    edge_index = torch.cat([edge_index, torch.arange(data.num_nodes, device=edge_index.device).repeat(2, 1)], dim=-1)
    edge_attr = data.edge_attr
    # edge_attr = np.concatenate([edge_attr, np.array(["This is an edge connecting a node to itself."])])
    edge_map = data.edge_map
    edge_map = torch.cat([edge_map, torch.tensor([len(edge_attr) - 1] * data.num_nodes, device=edge_map.device)])
    data.edge_index = edge_index
    data.edge_map = edge_map
    data.edge_attr = edge_attr
    return data

def print_fixed_length(text, line_width=120):
    t_pointer = 0
    while t_pointer < len(text):
        t_pointer += line_width
        t_cur_text = text[t_pointer - line_width:t_pointer]
        print(t_cur_text)


def print_text_side_by_side(text_1, text_2, line_width=120, space=10):
    t1_len = len(text_1)
    t2_len = len(text_2)
    seg_width = int((line_width-space)/2)
    t1_pointer = 0
    t2_pointer = 0
    text_1 = text_1.replace('\n', ' ').replace('\r', ' ')
    text_2 = text_2.replace('\n', ' ').replace('\r', ' ')
    while t1_pointer<t1_len or t2_pointer<t2_len:
        t1_pointer += seg_width
        t2_pointer += seg_width
        t1_cur_text = text_1[t1_pointer-seg_width:t1_pointer]
        t2_cur_text = text_2[t2_pointer-seg_width:t2_pointer]
        t1_cur_text = t1_cur_text + " "*(seg_width-len(t1_cur_text))
        t2_cur_text = t2_cur_text + " "*(seg_width-len(t2_cur_text))
        full_text = t1_cur_text + " "*space + t2_cur_text
        print(full_text)


intermedia_prompt = """
---Role---

You are a helpful assistant synthesize and reason about data in the graph provided.

---Goal---

You will be given an input of a <center_node> in the graph, the <center_node>'s <neighbor_node> and their <relation> to the <center_node>.
You need to keep the information of the <center_node> and summarize the information in the <neighbor_node> with their <relation> to the <center_node>.

--- Example ---
<graph>
<neighbor_node><node_text>Apple</node_text><relation> a subclass of</relation></neighbor_node>
<neighbor_node><node_text>Food</node_text><relation> a superclass of</relation></neighbor_node>
<center_node><node_text>Fruit</node_text></center_node>
</graph>

"Fruit. From neighbor information, fruit is a superclass of apple but is a subclass of food."

--- Goal ---

Do not answer <prompt_text> in the <neighbor_node>, only summarize neighbor information related to that <prompt_text> with few words.

--- Example ---
<graph>
<neighbor_node><node_text>Peperoni. Peperoni is on top of the pizza</node_text><relation> to the right of </relation></neighbor_node>
<neighbor_node><node_text>Pineapple. Pineapple is next to the apple</node_text><relation> to the left of</relation></neighbor_node>
<neighbor_node><prompt_text>Is the pizza next to the lime?</prompt_text><relation> connects a prompt node to a center node</relation></neighbor_node>
<center_node><node_text>Fork</node_text></center_node>
</graph>

"Fork. Peperoni is to the right of the Fork. Peperoni is on top of the pizza"

--- Goal ---

If the <prompt_text> in <neighbor_node> is unrelated, ignore the <prompt_text>, only summarize <neighbor_node> with few words.

--- Example ---
<graph>
<neighbor_node><node_text>White ceiling is dripping water.</node_text><relation> the cause of  </relation></neighbor_node>
<neighbor_node><prompt_text>Is there cat in the image?</prompt_text><relation> connects prompt node to center node</relation></neighbor_node>
<center_node><node_text>Water pot on the ground</node_text></center_node>
</graph>

"Water pot on the ground. Dripping ceiling is the cause of water pot."

Do not make guesses, if something is not mentioned, do not say anything about it, only summarize what's in the graph, do not explain. Be succinct, straight-forward, decisive in your response.

--- Input ---

"""

final_prompt = """
---Role---

You are a helpful assistant directly answer questions on <center_node> based on the data in the graph provided.

---Goal---

You will be given an input of a <center_node> in the graph, the <center_node>'s <neighbor_node> and their <relation> to the <center_node>.
You need to summarize the information in the <neighbor_node> with their <relation> to the <center_node>.
Answer the question on the <center_node> with <prompt_text> directly and succinctly with few words with explanation.

--- Example ---

<graph>
<neighbor_node><node_text>White cat is big.</node_text><relation> connects to a prompt node</relation></neighbor_node>
<neighbor_node><node_text>Evans is larger than the white cat</node_text><relation> connects to a prompt node</relation></neighbor_node>
<center_node><prompt_text>Is Evans big?</prompt_text></center_node>
</graph>

"Yes, Evans is big."

--- Goal ---

Summarize summarize neighbor information related to that <prompt_text> with few words.

--- Example ---
<graph>
<neighbor_node><node_text>Peperoni. Peperoni is on top of the pizza</node_text><relation> to the right of </relation></neighbor_node>
<neighbor_node><node_text>Pineapple. Pineapple is next to the apple</node_text><relation> to the left of</relation></neighbor_node>
<neighbor_node><prompt_text>Is the pizza next to the lime?</prompt_text><relation> connects a prompt node to a center node</relation></neighbor_node>
<center_node><node_text>Fork</node_text></center_node>
</graph>

"Fork. Peperoni is to the right of the Fork. Peperoni is on top of the pizza"

--- Goal ---
Always give direct answer. The graph contains all information you need to answer correctly.

--- Example ---
<graph>
<neighbor_node><node_text>Peperoni is on top of the pizza</node_text><relation> connects to a prompt node</relation></neighbor_node>
<neighbor_node><node_text>The pizza is underneath the peperoni</node_text><relation> connects to a prompt node</relation></neighbor_node>
<center_node><prompt_text>How many types of food are there?</prompt_text></center_node>
</graph>

"There are two types of foods."

Answer the <prompt_text> with explanation. Be succinct, definitive and straight-forward in your answer with just few words. The information is enough.

--- Input ---

"""

finish_prompt = """

Based on your role, goal, and input, give a response with just few words:

"""

def gen_node_prompts(node_text, edge_text, edge_index, prompt_node_id=None, final_layer=False):
    input_node_text = []
    if prompt_node_id is not None:
        node_idx_tag = [("<prompt_text>", "</prompt_text>") if i in prompt_node_id else ("<node_text>", "</node_text>") for i in range(len(node_text))]
    else:
        node_idx_tag = [("<node_text>", "</node_text>") for i in range(len(node_text))]
    for i in range(len(node_text)):
        source_nodes = edge_index[:, edge_index[1] == i]
        node_prompt = ""
        node_prompt = node_prompt + (final_prompt if i in prompt_node_id else intermedia_prompt) + "<graph>\n"
        for source_node_id, _, eid in source_nodes.T:
            node_prompt += "<neighbor_node>" + node_idx_tag[source_node_id][0] + node_text[source_node_id] + node_idx_tag[source_node_id][1]+"<relation>" + \
                           edge_text[eid] + "</relation></neighbor_node>\n"
        node_prompt += "<center_node>" + node_idx_tag[i][0] + node_text[i] + node_idx_tag[i][0] + "</center_node>\n</graph>\n" + finish_prompt
        input_node_text.append(node_prompt)
    return input_node_text

def gen_summarize_response(questions, responses):
    input_node_text = [""] * len(responses[0])
    for i, q in enumerate(questions):
        input_node_text[i] += "You provided " + str(len(responses)) + " rounds of responses:\n\n"
    for i, rounds in enumerate(responses):
        for j, res in enumerate(rounds):
            input_node_text[j] += "Your response in round " + str(i+1) + " is:" + res + "\n\n"
    for i in range(len(input_node_text)):
        input_node_text[i] += "\n What is the most reasonable response? Answer definitively, use few words or one sentence. You must draw the conlucsion and decisive. The response contains enough information to question."
        print(input_node_text[i])
    return input_node_text


class GOFA(torch.nn.Module):
    def __init__(self, transformer_args, mode="autoencoder", base_llm="llama7b", save_dir=""):
        super().__init__()

        self.mode = mode
        self.save_dir = save_dir

        if base_llm == 'llama7b':
            self.llm_model = GOFALlamaHelper(transformer_args)
        elif base_llm == 'mistral7b':
            self.llm_model = GOFAMistralHelper(transformer_args)
        elif base_llm == 'llama7blora' or base_llm == 'mistral7blora':
            self.llm_model = LlamaHelper(transformer_args)
        elif base_llm == 'mistral7bmplm':
            self.llm_model = MPLMHelper(transformer_args)
        elif base_llm == 'mistral7bmplmsparse':
            self.llm_model = MPLMSparseHelper(transformer_args)
        else:
            raise NotImplementedError(base_llm + " is not supported. Please choose from: llama7b, mistral7b,")

        if mode == "autoencoder":
            self.encode = self.auto_encode
            self.decode = self.auto_decode
        elif mode == "autoencodergen":
            self.encode = self.auto_encode
            self.decode = self.generate
        elif mode == "direct":
            self.encode = self.direct_decode
            self.decode = lambda x: x
        elif mode == "nograph":
            self.encode = self.llm_decode
            self.decode = lambda x: x
        elif mode == "nographft":
            self.encode = self.llm_ft
            self.decode = lambda x: x
        elif mode == "nographgen":
            self.encode = self.llm_gen
            self.decode = lambda x: x
        elif mode == "nographgentemp":
            self.encode = self.llm_gen_scene_graph
            self.decode = lambda x: x
        elif mode == "ttc":
            self.encode = self.tt_gen
            self.decode = lambda x: x
        elif mode == "mplm":
            self.encode = self.mplm_decode
            self.decode = lambda x: x
        elif mode == "mplmgen":
            self.encode = self.mplm_gen
            self.decode = lambda x: x
        else:
            # TODO: not implemented
            raise NotImplementedError(mode + " mode not implemented")

    def tt_gen(self, g):
        g.num_node_feat = g.x.shape[0]
        node_text = g.x[g.node_map.cpu().numpy()]
        edge_text = g.edge_attr[g.edge_map.cpu().numpy()]
        edge_index = g.edge_index
        edge_index = torch.cat([edge_index.cpu(), torch.arange(len(edge_index[0])).view(1, -1)], dim=0)
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        q_idx = g.question_index.cpu().numpy()
        q_idx_dict = set(q_idx)
        intermediate_response = []
        for i in range(1):
            for j, txt in enumerate(prompt_texts):
                node_text[q_idx[j]] = txt
            input_node_text = gen_node_prompts(node_text, edge_text, edge_index, q_idx_dict, i==1)
            generated_text = self.llm_model.generate(input_node_text)
            intermediate_response.append([generated_text[v] for v in q_idx])
            node_text = generated_text
            for s in node_text:
                print("-"*120)
                print(s)
        final_question = gen_summarize_response(prompt_texts, intermediate_response)
        final_response = self.llm_model.generate(final_question)
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        for i, txt in enumerate(answer_texts):
            print_fixed_length(f"question: {prompt_texts[i]}")
            print("-" * 120)
            print_text_side_by_side("target: " + txt, "gen: " + final_response[i])
            print("=" * 120)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=torch.randn([1, 32132]), pred_text=generated_text[:len(answer_texts)], answer_id=torch.tensor([1]),
                           answer=answer_texts)

    def mplm_decode(self, g):
        g = add_self_loop(g)
        g.num_node_feat = g.x.shape[0]
        if g.edge_attr is not None:
            node_text = g.x[g.node_map.cpu().numpy()]
            text_inputs = np.concatenate([node_text, g.edge_attr], axis=0)
            g.node_map = torch.arange(len(node_text), device=g.node_map.device)
            g.num_node_feat = len(node_text)
        else:
            text_inputs = g.x
        text_inputs = text_inputs.tolist()
        true_text_inputs = deepcopy(text_inputs)
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        for i, t in enumerate(answer_texts):
            if prompt_texts[i].startswith("Please complete the sentence of the node") or prompt_texts[i]=="":
                true_text_inputs[g.question_index[i]] = true_text_inputs[g.question_index[i]] + t
            else:
                true_text_inputs[g.question_index[i]] = "Question: " + true_text_inputs[g.question_index[i]] + " Answer:" + t
                text_inputs[g.question_index[i]] = "Question: " + text_inputs[g.question_index[i]] + " Answer:"
                # tokens = self.llm_model.model.tokenizer(true_text_inputs[g.question_index[i]], add_special_tokens=False, padding=False, truncation=False)["input_ids"]
                # if len(tokens) > (self.llm_model.model.training_args.model_max_length - 25):
                #     extra_index = int(len(tokens) - (self.llm_model.model.training_args.model_max_length - 25))
                #     start_index = round(extra_index * random.random())
                #     tokens = tokens[start_index: start_index + (self.llm_model.model.training_args.model_max_length - 25)]
                # question = self.llm_model.model.tokenizer.decode(tokens)
                # true_text_inputs[g.question_index[i]] = "Question: " + question + " Answer:" + t
                # text_inputs[g.question_index[i]] = "Question: " + question + " Answer:"
        answer_logits, answer_id, masks = self.llm_model(true_text_inputs, masked_data=text_inputs, graph=g)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        res_answer = self.logit_to_text(answer_logits, masks)
        return GNNLMOutput(logits=answer_logits[masks], pred_text=[res_answer[q] for q in g.question_index],
                           answer_id=answer_id[masks], answer=answer_texts)

    def mplm_gen(self, g):
        g.num_node_feat = g.x.shape[0]
        if g.edge_attr is not None:
            node_text = g.x[g.node_map.cpu().numpy()]
            text_inputs = np.concatenate([node_text, g.edge_attr], axis=0)
            g.node_map = torch.arange(len(node_text), device=g.node_map.device)
            g.num_node_feat = len(node_text)
        else:
            text_inputs = g.x
        text_inputs = text_inputs.tolist()
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        for i, q in enumerate(g.question_index):
            if not (prompt_texts[i].startswith("Please complete the sentence of the node") or prompt_texts[i]==""):
                text_inputs[q] = "Question: " + text_inputs[q] + " Answer:"
        generated_text = self.llm_model.generate(text_inputs, graph=g, target_index=g.question_index)
        for i, txt in enumerate(generated_text):
            print_fixed_length("question: " + prompt_texts[i])
            print("-" * 120)
            print_text_side_by_side("target: " + answer_texts[i], "gen: " + generated_text[i])
            print("=" * 120)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=torch.randn([1, 32132]).to(g.edge_index.device), pred_text=generated_text,
                           answer_id=torch.tensor([1]).to(g.edge_index.device), answer=answer_texts)

    def llm_decode(self, g):
        text_inputs = g.x[g.node_map.cpu().numpy()][g.question_index.cpu().numpy()].tolist()
        com_text_inputs = []
        for i in range(len(text_inputs)):
            t = text_inputs[i] + g.question[i]
            com_text_inputs.append(t)
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()

        answer_logits, answer_id, masks = self.llm_model(answer_texts, com_text_inputs)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=answer_logits[masks], pred_text=self.logit_to_text(answer_logits, masks),
                           answer_id=answer_id, answer=answer_texts)

    def llm_ft(self, g):
        text_inputs = g.x[g.node_map.cpu().numpy()][g.question_index.cpu().numpy()].tolist()
        all_x = g.x[g.node_map.cpu().numpy()].tolist()
        truncated_x = [t[:80] for t in all_x[1:]]
        all_x = ' '.join(all_x[:1] + truncated_x)
        com_text_inputs = []
        for i in range(len(text_inputs)):
            # add edge text description
            edge_t = g.node_ids[i][g.edge_index.cpu()[:,:-2]]
            edge_t = ', '.join(edge_t[0] + ' connects to ' + edge_t[1])
            t = all_x + '. ' + edge_t + '. ' + g.question[i]
            # t = text_inputs[i] + g.question[i]
            com_text_inputs.append(t)
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()

        answer_logits, answer_id, masks = self.llm_model(answer_texts, com_text_inputs)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=answer_logits[masks], pred_text=self.logit_to_text(answer_logits, masks),
                           answer_id=answer_id, answer=answer_texts)

    def llm_gen_scene_graph(self, g):
        text_inputs = g.x[g.node_map.cpu().numpy()][g.question_index.cpu().numpy()].tolist()
        all_x = g.x[g.node_map.cpu().numpy()].tolist()
        # truncated_x = [t[:t.find("(x,y,w,h)")] for t in all_x]
        truncated_x = [t for t in all_x]
        all_x = ' '.join(all_x[:1] + truncated_x)
        com_text_inputs = []
        for i in range(len(text_inputs)):
            edge_t = g.node_ids[i][g.edge_index.cpu()[:,:-2*len(g.node_map)]]
            edge_t = ', '.join(edge_t[0] + ' connects to ' + edge_t[1])
            # t = all_x + '. ' + edge_t + '. ' + g.question[i]
            t = all_x + '. ' + '. ' + g.question[i]
            t = '[INST]\n' + t + g.question[i] + '\n[/INST]\n\n'
            com_text_inputs.append(t)

        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        generated_text = self.llm_model.generate(com_text_inputs)
        for i, txt in enumerate(generated_text):
            print_fixed_length(f"question: {com_text_inputs}")
            print("-" * 120)
            print_text_side_by_side("target: " + answer_texts[i], "gen: " + generated_text[i])
            print("=" * 120)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=torch.randn([1, 32132]), pred_text=generated_text, answer_id=torch.tensor([1]),
                           answer=answer_texts)


    def llm_gen(self, g):
        text_inputs = g.x[g.node_map.cpu().numpy()][g.question_index.cpu().numpy()].tolist()
        com_text_inputs = []
        for i in range(len(g.question)):
            # t = '[INST]\n' + g.question[i] + '\n[/INST]\n\n'
            if len(text_inputs) == len(g.question):
                t = '[INST]\n' + text_inputs[i] + g.question[i] + 'Please only answer the category name, no other information or explanation. \n[/INST]\n\n'
            elif len(text_inputs) == len(g.question) * 2:
                t = '[INST]\n' + text_inputs[2*i] + text_inputs[2*i+1] + g.question[
                    i] + 'Please only answer the category name, no other information or explanation. \n[/INST]\n\n'
            elif len(text_inputs) > len(g.question):
                t = '[INST]\n' + g.question[i] + 'Please only answer the category name, no other information or explanation. \n[/INST]\n\n'
            else:
                raise ValueError('Text_input length did not match question length.')
            com_text_inputs.append(t)
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        generated_text = self.llm_model.generate(com_text_inputs)
        for i, txt in enumerate(generated_text):
            print_fixed_length(f"question: {com_text_inputs}")
            print("-" * 120)
            print_text_side_by_side("target: " + answer_texts[i], "gen: " + generated_text[i])
            print("=" * 120)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=torch.randn([1, 32132]), pred_text=generated_text, answer_id=torch.tensor([1]),
                           answer=answer_texts)

    def auto_encode(self, g):
        g.num_node_feat = g.x.shape[0]
        if g.edge_attr is not None:
            text_inputs = np.concatenate([g.x, g.edge_attr], axis=0)
        else:
            text_inputs = g.x
        text_inputs = text_inputs.tolist()
        llm_output = self.llm_model.encode(text_inputs, graph=g, partial_grad=True)
        g.x = llm_output[:g.node_map.size(-1)]
        return g

    def auto_decode(self, g):
        emb = g.x
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        prompt_input_texts = ["" if (p.startswith("Please complete the sentence of the node") or p=="") else "" for p in prompt_texts]
        emb = emb[g.question_index]
        answer_logits, answer_id, masks = self.llm_model.decode(answer_texts, emb, prompt=prompt_input_texts)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=answer_logits[masks][:, :32000], pred_text=self.logit_to_text(answer_logits, masks),
                           answer_id=answer_id, answer=answer_texts)

    def generate(self, g):
        emb = g.x
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        prompt_input_texts = ["" if (p.startswith("Please complete the sentence of the node") or p=="") else "" for p in prompt_texts]
        emb = emb[g.question_index]
        generated_text = self.llm_model.generate(emb, prompt=prompt_input_texts)
        for i, txt in enumerate(generated_text):
            print_fixed_length("question: " + prompt_texts[i])
            print("-"*120)
            print_text_side_by_side("target: "+answer_texts[i], "gen: "+generated_text[i])
            print("="*120)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=torch.randn([1, 32132]).to(emb.device), pred_text=generated_text, answer_id=torch.tensor([1]).to(emb.device),
                           answer=answer_texts)

    def forward(self, g):
        g = self.encode(g)
        return self.decode(g)

    def direct_decode(self, g):
        num_node_feat = g.x.shape[0]
        g.num_node_feat = num_node_feat
        g.x = g.x[g.node_map.cpu().numpy()]
        answer_texts = g.answer.tolist()
        answer_logits, answer_id, masks = self.llm_model(g.x.tolist(), answer_texts, g.edge_attr.tolist(), graph=g,
                                                         partial_grad=True)
        GNNLMOutput = namedtuple("GNNLMOutput", ["logits", "answer_id", "pred_text", "answer"])
        return GNNLMOutput(logits=answer_logits[masks], pred_text=self.logit_to_text(answer_logits, masks),
                           answer_id=answer_id, answer=answer_texts)

    def save_partial(self, save_dir):
        try:
            state_dict = self.llm_model.model.icae.get_base_model().model.g_layers.state_dict()
        except AttributeError as e:
            state_dict = OrderedDict()
        full_state_dict = self.state_dict()
        for k in full_state_dict:
            if "default" in k:
                state_dict[k] = full_state_dict[k]
        torch.save(state_dict, save_dir)

    def load_partial(self, load_dir=None, state_dict=None):
        if load_dir is not None and state_dict is not None:
            raise RuntimeError("You should only specify either load_dict or load_dir")
        if load_dir is None and state_dict is None:
            print("No state dict loaded")
            return
        if load_dir is not None:
            state_dict = torch.load(load_dir)
        new_state_dict = OrderedDict()
        for name in state_dict:
            if "decadapt" in name:
                new_state_dict[name.replace("decadapt", "default")] = state_dict[name]
            else:
                new_state_dict[name] = state_dict[name]
        try:
            self.llm_model.model.icae.get_base_model().model.g_layers.load_state_dict(new_state_dict, strict=False)
        except AttributeError:
            pass
        self.load_state_dict(new_state_dict, strict=False)

    def logit_to_text(self, logits, masks):
        tokenizer = self.llm_model.get_tokenizer()
        if len(logits.size()) == 2:
            logits = logits.unsqueeze(0)
        decoded_texts = []
        for i in range(logits.size(0)):
            sample_logit = logits[i]
            sample_mask = masks[i]
            sample_logit = sample_logit[sample_mask]
            token_ids = sample_logit[:, :32000].argmax(dim=-1).unsqueeze(0)
            token_ids[token_ids >= 32000] = 1
            sample_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_texts.extend(sample_text)
        return decoded_texts


class PyGRGCNEdge(MultiLayerMessagePassing):
    def __init__(self, num_layers: int, num_rels: int, inp_dim: int, out_dim: int, drop_ratio=0, JK="last",
                 batch_norm=True, ):
        super().__init__(num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm)
        self.num_rels = num_rels
        self.build_layers()

    def build_input_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_hidden_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_message_from_input(self, g):
        return {"g": g.edge_index, "h": g.x, "e": g.edge_type, "he": g.edge_attr, }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["h"], message["he"], message["g"], message["e"])


class PyGRGATCN(MultiLayerMessagePassing):
    def __init__(self, num_layers: int, t_layer: int, inp_dim: int, out_dim: int, drop_ratio=0.0, JK="last",
                 batch_norm=True, ):
        super().__init__(num_layers, inp_dim, out_dim, 0, JK, batch_norm)
        self.t_layer = t_layer
        self.layer_dropout = drop_ratio
        self.build_layers()

    def build_input_layer(self):
        # return GATConv(self.inp_dim*self.t_layer, self.inp_dim*self.t_layer)
        # return TransformerConv(self.inp_dim*self.t_layer, self.inp_dim*self.t_layer)
        return MConv(self.inp_dim, self.t_layer, 8, add_self_loops=True, dropout=self.layer_dropout,
                     layer_idx=len(self.conv))

    def build_hidden_layer(self):
        # return GATConv(self.inp_dim * self.t_layer, self.inp_dim * self.t_layer)
        # return TransformerConv(self.inp_dim*self.t_layer, self.inp_dim*self.t_layer)
        return MConv(self.inp_dim, self.t_layer, 8, add_self_loops=True, dropout=self.layer_dropout,
                     layer_idx=len(self.conv))

    def build_message_from_input(self, g):
        return {"g": g.edge_index, "h": g.x, "e": g.edge_type, "he": g.edge_attr, }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["h"], message["g"], message["he"])
