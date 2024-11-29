import os
from datasets import load_dataset, Dataset
import joblib
import torch
import warnings
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

from collections import defaultdict


def predict(
    text_batch, model, tokenizer, platt_scaler, label_encoder, target_class="clean"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Concatenate all lines from all documents and keep track of their original order
    all_lines_with_idx = []
    for doc_idx, text in enumerate(text_batch):
        for line_idx, line in enumerate(text.splitlines()):
            all_lines_with_idx.append((len(line.split()), doc_idx, line_idx, line))

    # Sort lines by their length (number of words) for efficient padding
    all_lines_with_idx.sort(key=lambda x: x[0])

    # Process each group of lines by length
    all_scaled_probs = [None] * len(
        all_lines_with_idx
    )  # Pre-allocate space for results
    for i in tqdm(range(0, len(all_lines_with_idx), 128), desc="Processing text batch"):
        # Extract a batch of lines with their metadata
        batch = all_lines_with_idx[i : i + 128]
        line_batch = [x[3] for x in batch]  # Get the actual text lines

        # Tokenize the batch of lines with padding based on the longest line in the batch
        inputs = tokenizer(
            line_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()  # Move logits to CPU

        # Extract logits for the target class
        target_class_index = label_encoder.transform([target_class])[0]
        positive_logits = logits[:, target_class_index]

        # Apply Platt scaling on the logits
        scaled_probs = platt_scaler.predict_proba(positive_logits.reshape(-1, 1))[:, 1]

        # Store the scaled probabilities in the original order
        for j, (_, doc_idx, line_idx, _) in enumerate(batch):
            all_scaled_probs[i + j] = (doc_idx, line_idx, round(scaled_probs[j], 4))

    # Organize scaled probabilities back into the structure of the original text_batch
    doc_scaled_probs = [[] for _ in text_batch]
    for doc_idx, line_idx, prob in sorted(all_scaled_probs, key=lambda x: (x[0], x[1])):
        doc_scaled_probs[doc_idx].append(prob)

    return doc_scaled_probs


def run(model_name, model, tokenizer, label_encoder, target_class="clean"):
    platt_scaler = joblib.load(f"platt_scaler_{model_name}.joblib")
    model.half()
    model.eval()
    print(f"Using {torch.cuda.device_count()} GPUs")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Use streaming to load data row by row
    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", name="sample-10BT")

    # Set up parameters
    output_dir = "exquisiteweb"
    checkpoint_file = os.path.join(output_dir, "checkpoint.txt")
    shard_size = 10000  # Set a larger shard size for saving
    shard = []
    shard_idx = 0

    os.makedirs(output_dir, exist_ok=True)

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            shard_idx = int(f.read().strip())

    # Resume from the next shard if checkpoint exists
    current_row = 0
    for example in dataset:
        # If a checkpoint exists, skip processed rows
        if current_row < shard_idx * shard_size:
            current_row += 1
            continue

        # Accumulate rows until the shard size is reached
        shard.append(example)
        current_row += 1

        if len(shard) == shard_size:
            # Extract the 'text' field and process the lines
            text_batch = [item["text"] for item in shard]
            scaled_probs_batch = predict(
                text_batch, model, tokenizer, platt_scaler, label_encoder, target_class
            )

            # Store the scaled probabilities back in each item of the shard
            for item, scaled_probs in zip(shard, scaled_probs_batch):
                item["line_quality"] = scaled_probs

            # Convert shard to Dataset and save
            shard_dataset = Dataset.from_list(shard)
            shard_dataset.save_to_disk(os.path.join(output_dir, f"shard_{shard_idx}"))

            # Update the checkpoint file
            shard_idx += 1
            with open(checkpoint_file, "w") as f:
                f.write(str(shard_idx))

            print(f"Saved shard {shard_idx}")

            # Clear the shard to start the next one
            shard = []

    # Save any remaining rows as the final shard
    if shard:
        text_batch = [item["text"] for item in shard]
        scaled_probs_batch = predict(
            text_batch, model, tokenizer, platt_scaler, label_encoder, target_class
        )

        for item, scaled_probs in zip(shard, scaled_probs_batch):
            item["line_quality"] = scaled_probs

        shard_dataset = Dataset.from_list(shard)
        shard_dataset.save_to_disk(os.path.join(output_dir, f"shard_{shard_idx}"))

        with open(checkpoint_file, "w") as f:
            f.write(str(shard_idx + 1))