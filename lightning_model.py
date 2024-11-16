import os
from typing import Any, Optional, Dict, Union, Callable

import numpy as np

from gp.lightning.module_template import BaseTemplate
import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
# from dataclasses import dataclass, field
# from torch import Tensor
# import traceback
# from lightning.pytorch.loops.optimization.automatic import Closure
# from functools import partial

class GraphPredLightning(BaseTemplate):
    def forward(self, batch):
        return self.model(batch)

    def on_train_start(self) -> None:
        torch.cuda.empty_cache()
        self.optimizers().param_groups[0]['lr'] = self.exp_config.lr
        self.lr_schedulers().last_epoch = -1
        self.lr_schedulers().T_max = self.exp_config.T_max

def make_dummy_batch(batch):
    batch.edge_index = torch.zeros((2, 0), device=batch.edge_index.device, dtype=batch.edge_index.dtype)
    batch.question_map = torch.tensor([0], device=batch.question_map.device, dtype=batch.edge_index.dtype)
    batch.answer_map = torch.tensor([0], device=batch.question_map.device, dtype=batch.edge_index.dtype)
    batch.node_map = torch.tensor([0], device=batch.question_map.device, dtype=batch.edge_index.dtype)
    batch.edge_map = torch.tensor([], device=batch.question_map.device, dtype=batch.edge_index.dtype)
    batch.x = np.array(["Your name is "], dtype=object)
    batch.edge_attr = np.array([], dtype=object)
    batch.question = np.array(["Your name is "], dtype=object)
    batch.answer = np.array(["GOFA."],dtype=object)
    batch.question_index = torch.tensor([0], device=batch.question_map.device, dtype=batch.edge_index.dtype)

class GraphTextPredLightning(BaseTemplate):
    def forward(self, batch):
        # print(batch)
        return self.model(batch)

    def on_train_start(self) -> None:
        torch.cuda.empty_cache()
        self.optimizers().param_groups[0]['lr'] = self.exp_config.lr
        self.lr_schedulers().last_epoch = -1
        self.lr_schedulers().T_max = self.exp_config.T_max

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        pass

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.model.save_partial(os.path.join(self.model.save_dir, "mem_ckpt.pth"))
        for k in list(checkpoint["state_dict"].keys()):
            if "g_layers" not in k:
                del checkpoint["state_dict"][k]

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            score, loss = self.compute_results(batch, batch_idx, self.exp_config.train_state_name[dataloader_idx])
        except RuntimeError as e:
            if "out of memory" in str(e):
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                self.optimizers().zero_grad()
                torch.cuda.empty_cache()
                make_dummy_batch(batch)
                print("OOM batch use dummy")
                score, loss = self.compute_results(batch, batch_idx, self.exp_config.train_state_name[dataloader_idx])
            else:
                raise e
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.old_decode = self.model.decode
        self.model.decode = self.model.auto_generate

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self.model.decode = self.old_decode

    #
    #
    # def optimizer_step(
    #     self,
    #     epoch: int,
    #     batch_idx: int,
    #     optimizer: Union[Optimizer, LightningOptimizer],
    #     optimizer_closure: Optional[Callable[[], Any]] = None,
    # ) -> None:
    #     try:
    #         output = optimizer_closure()
    #         def dummy_closure():
    #             return output
    #         optimizer.step(dummy_closure)
    #     except Exception as e:
    #         if "out of memory" in str(e):
    #             print("Ignoring optimizer OOM batch")
    #             print(traceback.format_exc())
    #             make_dummy_batch(optimizer_closure._step_fn.args[0]["batch"])
    #             output = optimizer_closure()
    #             def dummy_closure():
    #                 return output
    #             optimizer.step(dummy_closure)
    #         else:
    #             raise e
