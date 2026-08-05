# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Trainer with simple profiling utilities.

This subclass of HuggingFace's ``Trainer`` measures:
1. Data loading latency (time between the end of the previous ``training_step`` and
   the start of the current ``training_step``).
2. Forward-pass latency (time spent inside the base ``training_step`` implementation,
   which essentially wraps the model's forward / loss computation).

The statistics are logged via ``self.log`` every ``profile_log_interval`` steps and
also sent to the standard ``logging`` logger.  This is *not* meant to be a fully
fledged profiler – it is a quick, lightweight way to confirm whether the training
pipeline is bottlenecked by data loading or by the model's computation.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Optional

import torch
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class _BatchIterator:
    """Lightweight iterator that yields pre-collated batches."""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: Optional[int] = None
) -> torch.Tensor:  # noqa: D401
    """Compute token-level accuracy, ignoring ``-100`` label positions.

    Args:
        preds: Predicted token ids of shape ``(batch, seq_len)``.
        labels: Ground-truth label ids with the same shape as ``preds``.

    Returns:
        Scalar tensor with the fraction of correctly predicted labels in the
        current batch.
    """
    # casual prediction
    # Shift so that tokens < n predict n
    # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L60
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


class RouterFreezeDelayCallback(TrainerCallback):
    """Withhold condition-router logit updates for the first N optimizer steps.

    A uniform-init router at high LR co-adapts against a still-random DiT
    ("students choosing teachers before they can read"). Dropping the logit
    grads pre-optimizer-step keeps the logits in the optimizer's router group
    (unlike requires_grad=False, which would exclude them at create_optimizer
    time) and releases them exactly at ``freeze_steps``.
    """

    def __init__(self, logit_params: list, freeze_steps: int):
        self.logit_params = logit_params
        self.freeze_steps = freeze_steps
        self._released = False

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step < self.freeze_steps:
            for p in self.logit_params:
                p.grad = None
        elif not self._released:
            logging.warning(
                f"router-freeze-delay: logits released at global_step={state.global_step}"
            )
            self._released = True


class Gr00tTrainer(Trainer):
    """Trainer that bypasses torch dataloader and makes data collator async."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # noqa: D401 – simple description above
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        super().__init__(*args, **kwargs)
        freeze_steps = int(getattr(self.model.config, "router_freeze_steps", 0) or 0)
        if freeze_steps > 0:
            logit_params = [
                p
                for n, p in self.model.named_parameters()
                if "condition_router" in n and n.endswith("logits")
            ]
            if logit_params:
                self.add_callback(RouterFreezeDelayCallback(logit_params, freeze_steps))
                logging.warning(
                    f"router-freeze-delay: logits frozen for first {freeze_steps} optimizer steps"
                )

    def create_optimizer(self):
        """HF default optimizer, plus a dedicated param group for condition-router
        params (name contains 'condition_router') at model.config.router_lr,
        weight_decay 0. Covers both the plain and DeepSpeed (no-optimizer-in-
        ds-config) paths."""
        opt_model = self.model_wrapped if self.model_wrapped is not None else self.model

        def is_backbone(name):  # wrapper-prefix tolerant (e.g. DeepSpeed module.)
            return name.split("module.")[-1].startswith("backbone.")

        # Router group = gate logits + per-candidate norms. The optional
        # per-candidate proj adapters (".projs.") are representation weights,
        # not routing — they train at base lr with normal decay treatment.
        router_named = [
            (n, p)
            for n, p in opt_model.named_parameters()
            if "condition_router" in n and "projs." not in n and p.requires_grad
        ]
        backbone_lr = getattr(opt_model.config, "backbone_lr", None)
        backbone_named = (
            [
                (n, p)
                for n, p in opt_model.named_parameters()
                if is_backbone(n) and "condition_router" not in n and p.requires_grad
            ]
            if backbone_lr is not None
            else []
        )
        if self.optimizer is not None or (not router_named and not backbone_named):
            return super().create_optimizer()

        from transformers.trainer import Trainer as HFTrainer

        special = {n for n, _ in router_named} | {n for n, _ in backbone_named}
        decay_parameters = set(self.get_decay_parameter_names(opt_model))
        router_lr = getattr(opt_model.config, "router_lr", None) or self.args.learning_rate
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n in decay_parameters and n not in special and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n not in decay_parameters and n not in special and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        if backbone_named:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in backbone_named if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": backbone_lr,
                },
                {
                    "params": [p for n, p in backbone_named if n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": backbone_lr,
                },
            ]
        if router_named:
            optimizer_grouped_parameters.append(
                {
                    "params": [p for _, p in router_named],
                    "weight_decay": 0.0,
                    "lr": router_lr,
                }
            )
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]
        optimizer_cls, optimizer_kwargs = HFTrainer.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        # ZeRO CPU-offload requires DeepSpeed's AVX CPU Adam (torch AdamW on
        # offloaded fp32 states is ~10x slower per step and DeepSpeed rejects
        # it by default via zero_force_ds_cpu_optimizer).
        if os.environ.get("GROOT_DS_OFFLOAD") == "cpu":
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer_cls = DeepSpeedCPUAdam
            optimizer_kwargs = {
                k: v for k, v in optimizer_kwargs.items() if k in ("lr", "betas", "eps", "weight_decay")
            }
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        logging.info(
            f"create_optimizer: router group {len(router_named)} params at lr={router_lr}; "
            f"backbone group {len(backbone_named)} params at lr={backbone_lr}"
        )
        return self.optimizer

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Hide epoch from logged metrics as it's misleading for Iterable datasets.
        epoch = self.state.epoch
        self.state.epoch = None
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch

    def get_train_dataloader(self):  # noqa: D401
        """Return a iterable dataloader without skipping the data during resume, but reseed the dataset instead."""

        # Fall back to default behaviour if not using the custom buffer.
        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            # ``new_seed`` MUST be the same on every rank: ``ShardedMixtureDataset``
            # builds its shard schedule from this seed and partitions disjointly
            # by index, so a per-rank delta here would cause sample duplication
            # / loss across ranks. Both inputs are rank-symmetric (the dataset's
            # own seed was set rank-symmetrically at __init__, and global_step
            # is read from TrainerState which is broadcast via rendezvous).
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )
        # Use persistent workers for sharded dataset if num_workers is greater than 0
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        # multiprocessing_context can only be used with num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Pre-load TrainerState before super().train() so get_train_dataloader
        can read self.state.global_step (stateful samplers rely on this).
        ``resume_from_checkpoint=True`` with no checkpoint raises rather than
        silently starting fresh.
        """
        if resume_from_checkpoint is True:
            latest_checkpoint = get_last_checkpoint(self.args.output_dir)
            if latest_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )
        elif resume_from_checkpoint in (False, None):
            latest_checkpoint = None
        else:
            latest_checkpoint = resume_from_checkpoint  # caller passed an explicit path

        if latest_checkpoint is not None:
            logging.info(f"Resuming from checkpoint {latest_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(latest_checkpoint, TRAINER_STATE_NAME)
            )

        return super().train(resume_from_checkpoint=latest_checkpoint, **kwargs)

    # ------------------------------------------------------------------
    # Loss / accuracy computation override
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """Compute loss *and* log token-level accuracy every training step.

        We delegate the heavy-lifting (including label smoothing, custom loss
        functions, etc.) to the parent ``Trainer.compute_loss`` implementation
        by calling it with ``return_outputs=True``.  After obtaining the loss
        *and* model outputs, we calculate accuracy and push it to the logger.
        """

        # Use parent implementation to preserve built-in functionality.
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        # import ipdb; ipdb.set_trace()
        # # save the model's embedding for the first step
        # input_embeddings = model.get_input_embeddings().weight.data.cpu()
        # output_embeddings = model.get_output_embeddings().weight.data.cpu()
        # torch.save(input_embeddings, f"input_embeddings_{self.state.global_step}.pt")
        # torch.save(output_embeddings, f"output_embeddings_{self.state.global_step}.pt")

        # Record last loss for testing purposes.
        self.loss = loss

        # Entropy-annealed router exploration: loss -= coef(t) * mean routing
        # entropy, coef(t) linear to 0 over max_steps. The static router's
        # entropy depends only on the logits, so no forward-pass plumbing.
        coef0 = float(getattr(self.model.config, "router_entropy_coef", 0.0) or 0.0)
        if coef0 > 0.0 and model.training:
            m = model.module if hasattr(model, "module") else model
            router = getattr(getattr(m, "action_head", None), "condition_router", None)
            if router is not None and router.logits.requires_grad:
                anneal = max(0.0, 1.0 - self.state.global_step / max(1, self.args.max_steps))
                w = router.logits.softmax(dim=-1)
                entropy = -(w * (w + 1e-9).log()).sum(dim=-1).mean()
                loss = loss - coef0 * anneal * entropy
                self.loss = loss

        # --------------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and "labels" in inputs
        ):
            if self.action_offset is not None:
                preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
            else:
                preds = outputs.logits.detach().argmax(dim=-1).cpu()
            with torch.no_grad():
                acc_local = _batch_accuracy(
                    preds, inputs["labels"].to(device=preds.device), self.action_offset
                )
            acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
            acc_mean = self._nested_gather(acc_tensor).mean().item()

            if self.args.local_rank in (-1, 0):
                self.log({"train_accuracy": acc_mean})

                # Log a sample of ground-truth vs predicted action tokens from
                # the first batch element so users can verify the model is
                # learning the right behaviors.
                shifted_labels = inputs["labels"][:1, 1:].cpu()
                shifted_preds = preds[:1, :-1]
                mask_0 = shifted_labels[0] != -100
                gt_tokens = shifted_labels[0][mask_0][:20]
                if self.action_offset is not None:
                    gt_tokens = gt_tokens - self.action_offset
                gt_sample = gt_tokens.tolist()
                pred_sample = shifted_preds[0][mask_0[: shifted_preds.shape[1]]][:20].tolist()
                logging.info(
                    "Step %d — GT vs Pred (first 20 action tokens, batch[0]):\n"
                    "  GT:   %s\n  Pred: %s",
                    self.state.global_step,
                    gt_sample,
                    pred_sample,
                )

        # --------------------------------------------------------------
        # Condition-router mixture logging (RouterLLM/*, iron_vla TB naming)
        # --------------------------------------------------------------
        if not getattr(self, "_router_probe_done", False):
            self._router_probe_done = True
            logging.warning(
                "router-log probe: gs=%s logging_steps=%s local_rank=%s training=%s has_router=%s",
                self.state.global_step,
                self.args.logging_steps,
                self.args.local_rank,
                getattr(model, "training", "?"),
                getattr(getattr(self.model, "action_head", None), "condition_router", None) is not None,
            )
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and self.args.local_rank in (-1, 0)
        ):
            try:
                # Read stats straight from the module: mixture_stats() is
                # input-independent (softmax of the logits Parameter), and the
                # forward returns a BatchFeature (UserDict, NOT a dict subclass)
                # whose extra keys are awkward to reach generically.
                router = getattr(getattr(self.model, "action_head", None), "condition_router", None)
                if router is not None:
                    stats = router.mixture_stats()
                    rw = stats["router_weights"]
                    rent = stats["router_entropy"]
                    cfg = self.model.config
                    cands = getattr(cfg, "router_candidate_layers", None) or list(
                        range(getattr(cfg, "select_layer", 12) + 1)
                    )
                    w = rw.detach().float().cpu()  # [num_cross_blocks, K]
                    logs = {
                        f"RouterLLM/w_mean_L{layer:02d}": w[:, k].mean().item()
                        for k, layer in enumerate(cands)
                    }
                    # per-block incumbent extremes: which block leaves the stock tap first
                    logs["RouterLLM/w_incumbent_min"] = w[:, -1].min().item()
                    logs["RouterLLM/w_incumbent_mean"] = w[:, -1].mean().item()
                    if rent is not None:
                        logs["RouterLLM/entropy"] = float(rent)
                    self.log(logs)
            except Exception:  # diagnostics must never kill a training step
                logging.warning("router-stats logging failed", exc_info=True)

        return (loss, outputs) if return_outputs else loss
