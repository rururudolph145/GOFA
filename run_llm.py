import argparse
import os
from collections import OrderedDict
from datetime import timedelta

import shutil
from lightning.pytorch.loggers import WandbLogger
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from gp.utils.utils import (load_yaml, combine_dict, merge_mod, setup_exp, set_random_seed, )
from gp.lightning.metric import (EvalKit, )
from gp.lightning.data_template import DataModule
from gp.lightning.training import lightning_fit, lightning_test
from gp.lightning.module_template import ExpConfig
from lightning_model import GraphPredLightning, GraphTextPredLightning
from gofa_models.model import GOFA
from gofa_models.config import GOFALlamaConfig, GOFAMistralConfig

from torchmetrics import AUROC, Accuracy, MeanMetric, MeanAbsoluteError, Perplexity
from utils import (MultiApr, MultiAuc, SimAnyAuc, normalized_loss_factory, sentence_base, sentence_perplexity, mistral_binary_auc)
from gp.lightning.data_template import DataWithMeta
from tasks import GOFAPretrainTaskWrapper, GOFAFineTuneTaskWrapper
from TAGLAS.data import TAGData
from TAGLAS import get_evaluators
from TAGLAS.evaluation.interface import Evaluator
import torch
from types import SimpleNamespace
from functools import partial


def main(params):
    if params.base_llm.startswith('llama7b'):
        from modules.gofa_icae_llama_modeling import ModelArguments, TrainingArguments
        gofa_config = GOFALlamaConfig
    elif params.base_llm.startswith('mistral7b'):
        from modules.gofa_icae_mistral_modeling import ModelArguments, TrainingArguments
        gofa_config = GOFAMistralConfig
    else:
        raise NotImplementedError(params.base_llm + " is not supported. Please choose from: llama7b, mistral7b,")

    wandb_logger = WandbLogger(project=params.log_project, name=f"{params.exp_name}_{params.llm_name}",
                               save_dir=params.exp_dir, offline=params.offline_log, )
    print("available devices: ", torch.cuda.device_count())
    checkpoint_dir = os.path.join(params.exp_dir, params.log_project)
    params_dict = vars(params)
    wandb_logger.log_table(key="hparams", columns=list(params_dict.keys()), data=[list(params_dict.values())])
    model_args, training_args, gofa_args = ModelArguments(), TrainingArguments(), gofa_config(
        num_layers=params.num_layers)
    model_args.dec_lora = params.dec_lora
    model_args.llama_pretrain_checkpoint = params.llama_pretrain_checkpoint
    model_args.mistral_pretrain_checkpoint = params.mistral_pretrain_checkpoint
    training_args.model_max_length = params.llm_max_length
    if params.training_precision == "bf16-mixed":
        training_args.bf16 = True
        gofa_args.llama_dtype = torch.bfloat16
    gofa_args.gnn_mlp_type = params.mlp_type


    if params.run_mode == "pretrain":
        ######################################################################################################
        #                                          Pretrain Task                                             #
        ######################################################################################################
        task_names = ["mag240m", "arxiv", "pubmed_node", "wiki_graph", "wikikg90m"]

        save_names = ["pretrain_0"] * 5

        train_task = GOFAPretrainTaskWrapper(["arxiv"],
                                             root=params.data_root_path, save_name=f"pretrain_0", fast_data_load=True, single_node_cs=True)
        # train_task = GOFAPretrainTaskWrapper(["mag240m", "mag240m", "mag240m"], root=params.data_root_path,
        #                                     save_name=["pretrain_0", "pretrain_1", "pretrain_2"], fast_data_load=True, single_node_cs=True, from_saved=True)
        # train_task = GOFAPretrainTaskWrapper(["mag240m"], root=params.data_root_path,
        #                                      save_name=["pretrain_0"], fast_data_load=True,
        #                                      single_node_cs=True, from_saved=True)
        # train_task = GOFAPretrainTaskWrapper(task_names, root=params.data_root_path, save_name=save_names, fast_data_load=True, single_node_cs=True)
        val_tasks = GOFAPretrainTaskWrapper(["cora", "wikics", "products"], root=params.data_root_path, sample_size=3000,
                                          split="all", num_workers=params.num_workers, single_node_cs=True, from_saved=False)
        test_tasks = GOFAPretrainTaskWrapper(["cora"], root=params.data_root_path,sample_size=3000,
                                          split="all", num_workers=params.num_workers, single_node_cs=True)
        # val_tasks = GOFAPretrainTaskWrapper(["cora"], root=params.data_root_path,
        #                                     sample_size=1000,
        #                                     split="all", num_workers=params.num_workers, single_node_cs=True,
        #                                     from_saved=False)
        # test_tasks = GOFAPretrainTaskWrapper(["products"], root=params.data_root_path, sample_size=1000,
        #                                      split="all", num_workers=params.num_workers, single_node_cs=True, from_saved=False)

        # test spd
        # val_tasks = GOFAPretrainTaskWrapper(["cora"], root=params.data_root_path, split="all", sample_size=100,
        #                                     pretrain_tasks=["SP"], num_workers=params.num_workers,
        #                                     num_SP=1, from_saved=False, save_data=False, SP_from_targets=False)
        #
        # test_tasks = GOFAPretrainTaskWrapper(["cora"], root=params.data_root_path, split="all", sample_size=100,
        #                                      pretrain_tasks=["SP"], num_workers=params.num_workers,
        #                                      num_SP=1, from_saved=False, save_data=False, SP_from_targets=True)

        # breakpoint()

        n_steps = int(len(train_task) * params.num_epochs / (params.grad_acc_step * int(torch.cuda.device_count())))

        train_task = DataWithMeta(train_task, batch_size=params.batch_size, sample_size=params.train_sample_size)

        val_tasks = [DataWithMeta(val_tasks, batch_size=params.batch_size, sample_size=params.eval_sample_size,
                                state_name="val", metric="perp", classes=32132,
                                meta_data={"eval_func": sentence_perplexity})]

        test_tasks = [DataWithMeta(test_tasks, batch_size=params.batch_size, sample_size=params.eval_sample_size,
                                 state_name="test", metric="perp", classes=32132,
                                 meta_data={"eval_func": sentence_perplexity})]

        # val_tasks = [DataWithMeta(val_tasks, batch_size=params.batch_size, sample_size=params.eval_sample_size,
        #                         state_name="val", metric="text_mse", classes=32132,
        #                         meta_data={"eval_func": sentence_base})]
        # test_tasks = [DataWithMeta(test_tasks, batch_size=params.batch_size, sample_size=params.eval_sample_size,
        #                           state_name="test", metric="text_mse", classes=32132,
        #                           meta_data={"eval_func": sentence_base})]
        evlter = []
        # evlter = [Evaluator("text_mse"), Evaluator("text_mse")]


    else:
        train_tasks = params.train_task_names
        eval_tasks = params.eval_task_names

        if params.run_mode == "ft":
            ######################################################################################################
            #                                          FINETUNE Task                                             #
            ######################################################################################################
            def data_size_filter(data: TAGData, **kwargs):
                estimated_mem = 24.495 + 0.4645 * len(data.node_map) + 0.0042 * len(
                    torch.unique(data.node_map)) + 0.1689 * len(data.edge_map) + 0.2846 * len(torch.unique(data.edge_map))
                if len(data.node_map)+len(torch.unique(data.edge_map)) < 42 and estimated_mem < 73:
                    return data
                else:
                    return None
        else:
            ######################################################################################################
            #                                          Inference                                                 #
            ######################################################################################################
            def data_size_filter(data: TAGData, **kwargs):
                return data



        train_task = GOFAFineTuneTaskWrapper(train_tasks,
                                            root=params.data_root_path,
                                            split="train",
                                            hop=params.hops,
                                            max_nodes_per_hop=params.train_max_nodes_per_hops,
                                            sample_size=params.sample_size_per_task,
                                            post_funcs=data_size_filter,
                                            way=params.ways,
                                            num_workers=params.num_workers,
                                            instruction=params.instructs,
                                            selection=params.selections,
                                            add_prompt_graph=False,
                                            save_data=False,
                                            from_saved=False,)

        n_steps = int(len(train_task) * params.num_epochs / (params.grad_acc_step * int(torch.cuda.device_count())))

        # val_tasks = [OFAPretrainTaskWrapper(["arxiv"], root=params.data_root_path,
        #                                   split="val", data_multiple=10, k=1, num_workers=params.num_workers, csp_task=partial(CNTask, graph_text=True), from_saved=False)]
        # test_tasks = [OFAPretrainTaskWrapper(["arxiv"], root=params.data_root_path,
        #                                   split="test", data_multiple=1000, k=1, num_workers=params.num_workers, csp_task=partial(CNTask, graph_text=True), from_saved=False)]

        val_tasks = [GOFAFineTuneTaskWrapper(task_name,
                                            root=params.data_root_path,
                                            split="val",
                                            hop=hop,
                                            max_nodes_per_hop=max_nodes_per_hop,
                                            num_workers=params.num_workers,
                                            sample_size=inf_sample_size,
                                            way=way,
                                            instruction=instruct,
                                            selection=selection,
                                            add_prompt_graph=False,
                                            from_saved=False,
                                            save_data=False,
                                            ) for task_name, hop, max_nodes_per_hop, way, instruct, selection, inf_sample_size in
                                            zip(eval_tasks, params.inf_hops, params.inf_max_nodes_per_hops,
                                                params.inf_ways, params.inf_instructs, params.inf_selections, params.inf_sample_size_per_task)]

        test_tasks = [GOFAFineTuneTaskWrapper(task_name,
                                            root=params.data_root_path,
                                            split="test",
                                            hop=hop,
                                            max_nodes_per_hop=max_nodes_per_hop,
                                            num_workers=params.num_workers,
                                            sample_size=inf_sample_size,
                                            way=way,
                                            instruction=instruct,
                                            selection=selection,
                                            add_prompt_graph=False,
                                            from_saved=False,
                                            save_data=False,
                                            ) for task_name, hop, max_nodes_per_hop, way, instruct, selection, inf_sample_size in
                                            zip(eval_tasks, params.inf_hops, params.inf_max_nodes_per_hops,
                                                params.inf_ways, params.inf_instructs, params.inf_selections, params.inf_sample_size_per_task)]


        eval_metric_names, evaluators = get_evaluators(eval_tasks, task_types="QA")
        evlter = evaluators + evaluators

        # TODO: For Protein
        evlter = [AUROC(task="binary"), AUROC(task="binary")]
        eval_metric_names = ["roc_auc"]

        train_task = DataWithMeta(train_task, batch_size=params.batch_size, sample_size=params.train_sample_size)
        val_tasks = [DataWithMeta(task, batch_size=params.batch_size, sample_size=params.eval_sample_size,
                                  state_name=task_name + "_val", metric=metric_name, classes=32132,
                                  meta_data={"eval_func": mistral_binary_auc}) for task_name, task, metric_name in
                     zip(eval_tasks, val_tasks, eval_metric_names)]

        test_tasks = [DataWithMeta(task, batch_size=params.batch_size, sample_size=params.eval_sample_size,
                                   state_name=task_name + "_test", metric=metric_name, classes=32132,
                                   meta_data={"eval_func": mistral_binary_auc}) for task_name, task, metric_name in
                      zip(eval_tasks, test_tasks, eval_metric_names)]

        # train_task = DataWithMeta(train_task, batch_size=params.batch_size, sample_size=params.train_sample_size)
        # val_tasks = [DataWithMeta(task, batch_size=params.batch_size, sample_size=params.eval_sample_size,
        #                         state_name=task_name + "_val", metric=metric_name, classes=32132,
        #                         meta_data={"eval_func": sentence_base}) for task_name, task, metric_name in zip(eval_tasks, val_tasks, eval_metric_names)]
        #
        # test_tasks = [DataWithMeta(task, batch_size=params.batch_size, sample_size=params.eval_sample_size,
        #                         state_name=task_name + "_test", metric=metric_name, classes=32132,
        #                         meta_data={"eval_func": sentence_base}) for task_name, task, metric_name in zip(eval_tasks, test_tasks, eval_metric_names)]

    text_dataset = {"train": train_task, "val": val_tasks, "test": test_tasks}
    params.datamodule = DataModule(text_dataset, num_workers=params.num_workers)

    model = GOFA(transformer_args=[model_args, training_args, gofa_args], mode=params.mode, base_llm=params.base_llm)
    train_params = list(model.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=params.lr, weight_decay=params.l2, betas=(0.9, 0.95))
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=params.lr*0.1)
    lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
    # lr_scheduler_config = None

    eval_data = text_dataset["val"] + text_dataset["test"]
    val_state = [dt.state_name for dt in text_dataset["val"]]
    test_state = [dt.state_name for dt in text_dataset["test"]]
    eval_state = val_state + test_state
    eval_metric = [dt.metric for dt in eval_data]
    eval_funcs = [dt.meta_data["eval_func"] for dt in eval_data]
    loss = torch.nn.CrossEntropyLoss()
    loss_func = normalized_loss_factory(params.batch_size, training_args.model_max_length)
    if len(evlter) == 0:
        for dt in eval_data:
            if dt.metric == "acc":
                evlter.append(Accuracy(task="multiclass", num_classes=dt.classes))
            elif dt.metric == "auc":
                evlter.append(AUROC(task="binary"))
            elif dt.metric == "apr":
                evlter.append(MultiApr(num_labels=dt.classes))
            elif dt.metric == "aucmulti":
                evlter.append(MultiAuc(num_labels=dt.classes))
            elif dt.metric.startswith("sim"):
                sim_metric = dt.metric.split("_")[1]
                evlter.append(SimAnyAuc(sim_metric))
            elif dt.metric == "loss":
                evlter.append(MeanMetric())
            elif dt.metric == "mae":
                evlter.append(MeanAbsoluteError())
            elif dt.metric == "perp":
                evlter.append(Perplexity(ignore_index=model.llm_model.model.tokenizer.pad_token_id))
            else:
                raise NotImplementedError("unknown evaluator")
    metrics = EvalKit(eval_metric, evlter, loss, eval_funcs, loss_func, eval_mode="max", exp_prefix="",
                      eval_state=eval_state, val_monitor_state=val_state[0], test_monitor_state=test_state[0], )

    exp_config = ExpConfig("", optimizer, lr_scheduler=lr_scheduler_config)
    exp_config.val_state_name = val_state
    exp_config.test_state_name = test_state
    pred_model = GraphTextPredLightning(exp_config, model, metrics)
    if params.load_model:
        print("-"*60+"LOADING"+"-"*60)
        if os.path.isdir(params.load_dir):
            prefix = "_forward_module.model.llm_model.model.icae.base_model.model.model.g_layers."
            state_dict = get_fp32_state_dict_from_zero_checkpoint(params.load_dir)
            partial_dict = OrderedDict()
            for s in state_dict:
                if s.startswith(prefix):
                    partial_dict[s[len(prefix):]] = state_dict[s]
            model.load_partial(state_dict=partial_dict)
        else:
            model.load_partial(load_dir=params.load_dir)

    strategy = "deepspeed_stage_2" if torch.cuda.device_count() > 1 else "auto"
    if params.run_mode == "inf":
        val_res, test_res = lightning_test(wandb_logger, pred_model, params.datamodule, metrics, params.load_dir,
                                           strategy=strategy)
    else:
        val_res, test_res = lightning_fit(wandb_logger, pred_model, params.datamodule, metrics, params.num_epochs+params.last_epochs,
                                          strategy=strategy, save_model=params.save_model["save"], load_best=False,
                                          reload_freq=1, test_rep=params.test_rep, val_interval=params.val_interval,
                                          grad_clipping=params.grad_clip, grad_acc_step=params.grad_acc_step,
                                          save_time=timedelta(hours=params.save_model["time"]), cktp_prefix="best_ckpt",
                                          precision=params.training_precision, top_k=params.save_model["top_k"], ckpt_path=params.ckpt_path, save_last=params.save_model["last"])
    if params.last_save:
        # model.save_partial(os.path.join(params.exp_dir, "best_ckpt.pth"))
        torch.save(model.state_dict(), os.path.join(params.exp_dir, "best_ckpt.pth"))
    # if os.path.exists(checkpoint_dir):
    #     shutil.rmtree(checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument("--override", type=str)

    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line", )

    params = parser.parse_args()
    configs = []
    configs.append(load_yaml(os.path.join(os.path.dirname(__file__), "configs", "llama_config.yaml")))

    if params.override is not None:
        override_config = load_yaml(params.override)
        configs.append(override_config)

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    mod_params["root_path"] = mod_params["root_path"] if mod_params["root_path"] else os.environ.get("GGAMA_ROOT_PATH")
    mod_params["data_root_path"] = mod_params["data_root_path"] if mod_params["data_root_path"] else os.environ.get("GGAMA_ROOT_DATA_PATH")
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    torch.set_float32_matmul_precision("high")
    print(params)
    main(params)
