from huggingface_hub import hf_hub_download
import io
import shutil
import os.path as osp
import numpy as np
import torch
import os
from collections import  namedtuple
from types import SimpleNamespace

def safe_download_hf_file(repo_id,
                     filename,
                     local_dir,
                     subfolder=None,
                     repo_type="dataset",
                     cache_dir=None,
                     ):
    target_dir = osp.join(local_dir, filename)
    if not osp.exists(local_dir):
        os.makedirs(local_dir)
    if not osp.exists(target_dir):
        hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename, repo_type=repo_type,
                        local_dir=local_dir, local_dir_use_symlinks=False, cache_dir=cache_dir, force_download=True)
        if subfolder is not None:
            shutil.move(osp.join(local_dir, subfolder, filename), osp.join(local_dir, filename))
            shutil.rmtree(osp.join(local_dir, subfolder))
    return target_dir


def textualize_graph(data):
    # mapping from object id to index
    nodes = []
    edges = []
    entities = data['node']
    relations = data['edge']
    question_id = data['question']
    complete_id = data['complete']
    target_question = []
    target_answer = []
    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ind = np.arange(len(entities))
    ind = np.stack([(ind/26).astype(int), (ind%26).astype(int)], axis=1)
    char_index = [ascii_uppercase[ind[i][0]] + ascii_uppercase[ind[i][1]] for i in range(len(ind))]
    for nid in entities:
        if int(nid) in question_id:
            node_attr = f'{entities[nid]}'
            target_question.append(entities[nid])
            target_answer.append(entities[nid])
        elif int(nid) in complete_id:
            node_attr = f"This is [NODEID.{char_index[int(nid)]}]." + f'{entities[nid]}'
            target_question.append("")
            target_answer.append(entities[nid])
        else:
            node_attr = f"This is [NODEID.{char_index[int(nid)]}]." + f'{entities[nid]}'
        nodes.append({'node_id': int(nid), 'node_attr': node_attr})
    target_id = question_id + complete_id
    for rel in relations:
        src = int(rel["source"])
        dst = int(rel["target"])
        edge_attr = rel["relation"]["content"]
        edges.append({'src': src, 'edge_attr': edge_attr, 'dst': dst})
    return nodes, edges, target_id, target_question, target_answer

def batch_unique_feature(features):
    unique_feature, feature_map = np.unique(features, return_inverse=True)
    feature_map = torch.from_numpy(feature_map).long()

    return unique_feature, feature_map


GOFAGraphInput = namedtuple("GOFAGraphInput", ["edge_index", "x", "edge_attr", "node_map", "edge_map", "question", "answer", "question_map", "answer_map", "question_index"])

def prepare_gofa_graph_input(graph, device=None):
    t_graph_node, t_graph_edge, target_id, target_question, target_answer = textualize_graph(graph)

    node_text = [None] * len(t_graph_node)
    for i in range(len(t_graph_node)):
        node_text[int(t_graph_node[i]["node_id"])] = t_graph_node[i]["node_attr"]

    edges = []
    edge_texts = []

    for i in range(len(t_graph_edge)):
        edge_texts.append(t_graph_edge[i]["edge_attr"])
        edges.append([t_graph_edge[i]["src"], t_graph_edge[i]["dst"]])

    edges = torch.tensor(edges).T
    node_text = np.array(node_text)
    edge_texts = np.array(edge_texts)
    target_question = np.array(target_question)
    target_answer = np.array(target_answer)

    unique_node_feature, node_map = batch_unique_feature(node_text)
    unique_edge_feature, edge_map = batch_unique_feature(edge_texts)
    unique_question_feature, q_map = batch_unique_feature(target_question)
    unique_target_feature, a_map = batch_unique_feature(target_answer)

    names = ["edge_index", "x", "edge_attr", "node_map", "edge_map", "question", "answer", "question_map", "answer_map", "question_index"]

    features = [edges, unique_node_feature, unique_edge_feature, node_map, edge_map, unique_question_feature, unique_target_feature, q_map, a_map, torch.tensor(target_id, dtype=torch.long)]
    features = [f.to(device) if isinstance(f, torch.Tensor) else f for f in features]
    f_names = {n: features[i] for i, n in enumerate(names)}

    return SimpleNamespace(**f_names)
