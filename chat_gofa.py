import torch

from modules.gofa import GOFAMistralConfig, TrainingArguments, ModelArguments

from modules.gofa import GOFAMistral
from modules.utils import prepare_gofa_graph_input, prepare_gofa_graph_input_from_pyg
import json

graph_type = "pyg"
device = torch.device("cuda")
if graph_type == "json":
    with open("sample_graph.json", "r") as f:
        graph = json.load(f)
    gofa_input_graph = prepare_gofa_graph_input(graph, device=device)
elif graph_type == "pyg":
    graph = torch.load("sample_graph_pyg.pth")
    gofa_input_graph = prepare_gofa_graph_input_from_pyg(graph, device=device)
else:
    raise ValueError("Unknown graph type")
model_args, training_args, gofa_args = ModelArguments(), TrainingArguments(), GOFAMistralConfig()
model_args.dec_lora = True
gofa = GOFAMistral((model_args, training_args, gofa_args))
# use gofa.load_pretrained() to automatically download pretrained checkpoint into the cache directory. Or specify the checkpoint path to load.
gofa.load_pretrained()
gofa.to(device)
print(gofa.generate(gofa_input_graph))
