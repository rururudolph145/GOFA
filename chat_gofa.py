from modules.gofa import GOFAMistralConfig, TrainingArguments, ModelArguments

from modules.gofa import GOFAMistral
from modules.utils import prepare_gofa_graph_input
import json

device = None
model_args, training_args, gofa_args = ModelArguments(), TrainingArguments(), GOFAMistralConfig()
model_args.dec_lora = True
gofa = GOFAMistral((model_args, training_args, gofa_args))
# use gofa.load_pretrained() to automatically download pretrained checkpoint into the cache directory. Or specify the checkpoint path to load.
gofa.load_pretrained()
gofa.to(device)
with open("sample_graph.json", "r") as f:
    graph = json.load(f)

gofa_input_graph = prepare_gofa_graph_input(graph, device=device)

print(gofa(gofa_input_graph))
