import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk

def main() -> None:
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()

    training_args = TrainingArguments(
        output_dir="./multi_gpu_output",
        num_train_epochs=2,
        per_device_train_batch_size=2,      # ← Controls GPU workload
        per_device_eval_batch_size=2,       # ← Controls GPU workload
        logging_steps=1,
        dataloader_num_workers=7,             # ← Parallel data loading
        report_to="none",
        ddp_backend="nccl",
        logging_first_step=True,
        log_level="debug",
        log_level_replica="debug",
        dataloader_pin_memory=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        use_fast=True,
        trust_remote_code=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = load_from_disk(dataset_path="/flash/project_462000963/users/tarkkaot/preprocessed/train4/")
    dataset['train'] = dataset['train'].select(range(1000))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator
    )
    print("Starting training")
    trainer.train()


if __name__ == "__main__":
    main()