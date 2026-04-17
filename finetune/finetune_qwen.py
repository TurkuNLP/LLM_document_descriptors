#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import logging
import os
from shlex import split
import time
from typing import Any

import torch  # type: ignore
import transformers  # type: ignore
from accelerate import PartialState  # type: ignore
from datasets import Dataset, DatasetDict, load_from_disk  # type: ignore
from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# --------------------------------------------------------------------------------------
# Distributed state
# --------------------------------------------------------------------------------------

STATE = PartialState()


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

def configure_logging(log_file: str) -> None:
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    if STATE.is_main_process:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_h = logging.FileHandler(log_file)
        file_h.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        stream_h = logging.StreamHandler()
        stream_h.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_h)
        logging.root.addHandler(stream_h)
    else:
        logging.root.setLevel(logging.ERROR)
        logging.root.addHandler(logging.NullHandler())

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def log_main(msg: str, *args: Any) -> None:
    if STATE.is_main_process:
        logging.info(msg, *args)


# --------------------------------------------------------------------------------------
# Dataset utilities
# --------------------------------------------------------------------------------------

def make_ratio_splits(
    dataset: Dataset,
    test_size: float = 0.10,
    eval_size: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    if test_size <= 0 or eval_size <= 0 or (test_size + eval_size) >= 1.0:
        raise ValueError("Require 0 < test_size, eval_size and test_size + eval_size < 1")

    train_eval = dataset.train_test_split(
        test_size=test_size + eval_size,
        seed=seed,
        shuffle=True,
    )
    test_valid = train_eval["test"].train_test_split(
        test_size=eval_size / (test_size + eval_size),
        seed=seed,
        shuffle=True,
    )

    return DatasetDict(
        {
            "train": train_eval["train"],
            "validation": test_valid["test"],
            "test": test_valid["train"],
        }
    )


def make_fixed_holdout_splits(
    dataset: Dataset,
    val_size: int = 50_000,
    test_size: int = 50_000,
    seed: int = 42,
) -> DatasetDict:
    total_holdout = val_size + test_size
    if total_holdout >= len(dataset):
        raise ValueError(
            f"val_size + test_size must be smaller than dataset size "
            f"({total_holdout} >= {len(dataset)})"
        )

    split_1 = dataset.train_test_split(
        test_size=total_holdout,
        seed=seed,
        shuffle=True,
    )
    train_ds = split_1["train"]
    holdout_ds = split_1["test"]

    split_2 = holdout_ds.train_test_split(
        test_size=test_size,
        seed=seed,
        shuffle=True,
    )

    return DatasetDict(
        {
            "train": train_ds,
            "validation": split_2["train"],
            "test": split_2["test"],
        }
    )


def add_length(example: dict[str, Any]) -> dict[str, Any]:
    example["length"] = len(example["input_ids"])
    return example


def validate_dataset_columns(dataset: DatasetDict) -> None:
    required = {"input_ids", "attention_mask"}
    optional = {"length"}
    train_cols = set(dataset["train"].column_names)
    missing_required = required - train_cols
    missing_optional = optional - train_cols
    if missing_required:
        raise ValueError(
            f"Dataset is missing required tokenized columns: {sorted(missing_required)}. "
            "Expected a pretokenized dataset with input_ids and attention_mask."
        )
    if missing_optional:
        log_main(
            "Dataset is missing optional columns: %s. Consider adding with --add-length-column",
            sorted(missing_optional),
        )


def add_length_column(dataset: DatasetDict, training_args: TrainingArguments) -> DatasetDict:
    has_length = "length" in dataset["train"].column_names
    if has_length:
        return dataset

    with training_args.main_process_first(desc="add length column"):
        dataset = DatasetDict(
            {
                split: ds.map(add_length, num_proc=os.cpu_count(), desc=f"Adding length to {split}")
                for split, ds in dataset.items()
            }
        )
    return dataset

def filter_by_length(dataset: DatasetDict, training_args: TrainingArguments, max_length: int) -> DatasetDict:
    if max_length is None:
        return dataset
    
    if max_length <= 0:
        log_main("Invalid max_length %d; skipping length-based filtering", max_length)
        return dataset
    
    if "length" in dataset["train"].column_names:
        with training_args.main_process_first(desc="filter by length"):
            return DatasetDict(
                {
                    split: ds.filter(lambda ex: ex["length"] <= max_length, num_proc=os.cpu_count(), desc=f"Filtering {split} by length")
                    for split, ds in dataset.items()
                }
        )
    else:
        log_main("No length column found; skipping length-based filtering. Give arg --add-length-column to add a length column to the dataset and enable filtering.")
        return dataset

def log_dataset_stats(dataset: DatasetDict) -> None:
    if not STATE.is_main_process:
        return

    log_main(
        "Dataset sizes - train: %d, validation: %d, test: %d",
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
    )

    if "length" in dataset["train"].column_names:
        lengths = dataset["train"]["length"]
        sorted_lengths = sorted(lengths)

        def pct(p: float) -> int:
            idx = min(len(sorted_lengths) - 1, max(0, int(len(sorted_lengths) * p / 100)))
            return sorted_lengths[idx]

        log_main("Train length stats:")
        log_main("  min=%d", sorted_lengths[0])
        log_main("  p50=%d", pct(50))
        log_main("  p90=%d", pct(90))
        log_main("  p95=%d", pct(95))
        log_main("  p99=%d", pct(99))
        log_main("  max=%d", sorted_lengths[-1])


# --------------------------------------------------------------------------------------
# Debug callbacks
# --------------------------------------------------------------------------------------

class TrainerDebugCallback(TrainerCallback):
    def __init__(self) -> None:
        self.last_t: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        log_main(
            "on_train_begin: max_steps=%s num_train_epochs=%s",
            state.max_steps,
            args.num_train_epochs,
        )
        self.last_t = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        log_main("on_epoch_begin: epoch=%s global_step=%s", state.epoch, state.global_step)

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step < 5:
            log_main("on_step_begin: global_step=%s", state.global_step)

    def on_substep_end(self, args, state, control, **kwargs):
        if state.global_step < 3:
            log_main("on_substep_end: global_step=%s", state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        dt = -1.0 if self.last_t is None else now - self.last_t
        self.last_t = now
        log_main(
            "on_step_end: global_step=%s epoch=%s step_time=%.2fs",
            state.global_step,
            state.epoch,
            dt,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        log_main("on_log: global_step=%s logs=%s", state.global_step, logs)

    def on_save(self, args, state, control, **kwargs):
        log_main("on_save: global_step=%s", state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        log_main("on_evaluate: global_step=%s metrics=%s", state.global_step, metrics)


class DebugTrainer(Trainer):
    def get_train_dataloader(self):
        log_main("get_train_dataloader: enter")
        dl = super().get_train_dataloader()
        log_main("get_train_dataloader: done")
        return dl

    def _inner_training_loop(self, *args, **kwargs):
        log_main("_inner_training_loop: enter")
        return super()._inner_training_loop(*args, **kwargs)


# --------------------------------------------------------------------------------------
# Training setup
# --------------------------------------------------------------------------------------

def supports_training_arg(name: str) -> bool:
    return name in inspect.signature(TrainingArguments.__init__).parameters


def get_training_args(args) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "run_name": args.run_id,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": False,
        "bf16": True,
        "fp16": False,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": False,
        "load_best_model_at_end": False,
        "report_to": "wandb" if args.use_wandb else "none",
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
    }

    if args.do_eval:
        if supports_training_arg("eval_strategy"):
            kwargs["eval_strategy"] = "steps"
        elif supports_training_arg("evaluation_strategy"):
            kwargs["evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = args.eval_steps
    else:
        if supports_training_arg("eval_strategy"):
            kwargs["eval_strategy"] = "no"
        elif supports_training_arg("evaluation_strategy"):
            kwargs["evaluation_strategy"] = "no"

    # Version-safe length grouping
    if args.group_by_length:
        if supports_training_arg("train_sampling_strategy"):
            kwargs["train_sampling_strategy"] = "group_by_length"
            kwargs["length_column_name"] = "length"
        elif supports_training_arg("group_by_length"):
            kwargs["group_by_length"] = True
            kwargs["length_column_name"] = "length"

    return TrainingArguments(**kwargs)


def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=args.lora_target_modules.split(","),
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Keep trainable adapter weights in bf16
    #for param in model.parameters():
    #    if param.requires_grad:
    #        param.data = param.data.to(torch.bfloat16)

    if STATE.is_main_process:
        model.print_trainable_parameters()

    return model, tokenizer


def maybe_setup_wandb(args) -> None:
    if not args.use_wandb:
        return

    if STATE.is_main_process:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.run_id)
        os.environ.setdefault("WANDB_LOG_MODEL", "false")


def load_and_prepare_dataset(args, training_args: TrainingArguments) -> DatasetDict:
    log_main("Loading dataset from %s", args.data_dir)
    
    data = load_from_disk(args.data_dir)

    if isinstance(data, DatasetDict):
        dataset = data
        required_splits = {"train", "validation", "test"}
        if not required_splits.issubset(dataset.keys()):
            raise ValueError(
                f"DatasetDict found but missing required splits. Got: {list(dataset.keys())}"
            )
    elif isinstance(data, Dataset):
        log_main("Loaded a single Dataset; creating splits.")
        if args.fast_holdout:
            dataset = make_fixed_holdout_splits(
                data,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=args.seed,
            )
        else:
            dataset = make_ratio_splits(
                data,
                test_size=args.test_size_ratio,
                eval_size=args.eval_size_ratio,
                seed=args.seed,
            )
    else:
        raise TypeError(f"Unsupported dataset type: {type(data)}")

    validate_dataset_columns(dataset)

    if args.add_length_column:
        log_main("Adding length column to dataset")
        dataset = add_length_column(dataset, training_args)
    
    dataset = filter_by_length(dataset, training_args, args.filter_by_length)
    
    if args.group_by_length and "length" not in dataset["train"].column_names:
        log_main("Warning: group_by_length enabled but no length column found. Skipping length grouping. Consider adding a length column with --add-length-column.")
        args.group_by_length = False
        
    if STATE.is_main_process:
        ex = dataset["train"][0]
        log_main("Example input length (in tokens): %d", len(ex["input_ids"]))

    log_dataset_stats(dataset)
    return dataset


def debug_first_batch(trainer: Trainer) -> None:
    log_main("About to build train dataloader")
    train_dl = trainer.get_train_dataloader()
    log_main("Train dataloader built")

    log_main("About to fetch first batch")
    t0 = time.time()
    first_batch = next(iter(train_dl))
    log_main(
        "Fetched first batch in %.2fs; keys=%s",
        time.time() - t0,
        list(first_batch.keys()),
    )

    for k, v in first_batch.items():
        if hasattr(v, "shape"):
            log_main(
                "Batch tensor %s shape=%s dtype=%s",
                k,
                tuple(v.shape),
                v.dtype,
            )


def train_model(model, training_args, dataset, tokenizer, data_collator, args):
    log_main("Creating Trainer...")
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[TrainerDebugCallback()],
    )
    log_main("Trainer created")

    if args.debug_first_batch:
        debug_first_batch(trainer)

    log_main("Calling trainer.train()...")
    trainer.train()
    log_main("trainer.train() returned")

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    return trainer


def evaluate_model(trainer: Trainer, dataset: DatasetDict) -> None:
    log_main("Running test evaluation...")
    results = trainer.evaluate(eval_dataset=dataset["test"])
    log_main("Test results: %s", results)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA on a tokenized dataset")

    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--do-eval", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="descriptor-model-finetune")

    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")

    parser.add_argument("--lora-target-modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--add-length-column", action="store_true")
    parser.add_argument("--filter-by-length", type=int, default=None,
                        help="If set, filter out training examples with length greater than this value")
    
    parser.add_argument("--debug-first-batch", action="store_true")

    parser.add_argument("--fast-holdout", action="store_true")
    parser.add_argument("--val-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=50_000)
    parser.add_argument("--test-size-ratio", type=float, default=0.10)
    parser.add_argument("--eval-size-ratio", type=float, default=0.05)

    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(args) -> None:
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()

    maybe_setup_wandb(args)

    model, tokenizer = build_model_and_tokenizer(args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = get_training_args(args)
    dataset = load_and_prepare_dataset(args, training_args)

    trainer = train_model(
        model=model,
        training_args=training_args,
        dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=args,
    )

    if args.do_eval:
        evaluate_model(trainer, dataset)


if __name__ == "__main__":
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = f"./finetuned_models/{args.run_id}"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.log_file is None:
        args.log_file = f"logs/{args.run_id}.log"

    configure_logging(args.log_file)

    log_main("=" * 60)
    log_main("Starting fine-tuning")
    for key, value in vars(args).items():
        log_main("%s: %s", key, value)

    log_main("torch version: %s", torch.__version__)
    log_main("CUDA/HIP device count: %s", torch.cuda.device_count())
    log_main("LOCAL_RANK: %s", os.environ.get("LOCAL_RANK", "N/A"))
    log_main("RANK: %s", os.environ.get("RANK", "N/A"))
    log_main("WORLD_SIZE: %s", os.environ.get("WORLD_SIZE", "N/A"))
    log_main("SLURM_JOB_ID: %s", os.environ.get("SLURM_JOB_ID", "N/A"))

    main(args)