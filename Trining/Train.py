import os
import math
import json
import torch
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict
from difflib import SequenceMatcher
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)


# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_DIR = "/teamspace/studios/this_studio/Receipt-Information-Extraction-Project/Dataset/SROIE Binarize"
MODEL_ID    = "microsoft/layoutlmv3-base"
OUTPUT_DIR  = "./models/layoutlmv3-sroie-model-bin"

LABELS   = ["O", "COMPANY", "DATE", "ADDRESS", "TOTAL"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}


# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def normalize_bbox(box, width, height):
    return [
        max(0, min(1000, int(1000 * (box[0] / width)))),
        max(0, min(1000, int(1000 * (box[1] / height)))),
        max(0, min(1000, int(1000 * (box[2] / width)))),
        max(0, min(1000, int(1000 * (box[3] / height)))),
    ]


def fuzzy_match(fragment, entity_string, is_address=False, threshold=0.8):
    """Fuzzy matches text to handle OCR typos and greedy overlaps."""
    if not entity_string or not fragment:
        return False

    # Direct substring match (fast path)
    if fragment in entity_string:
        return True

    fragment_words = fragment.split()
    entity_words   = entity_string.split()

    match_count = 0
    for fw in fragment_words:
        if any(SequenceMatcher(None, fw, ew).ratio() > threshold for ew in entity_words):
            match_count += 1

    # Addresses require ≥50% word match to prevent stray numbers triggering it
    if is_address:
        return len(fragment_words) > 0 and (match_count / len(fragment_words)) >= 0.5

    # Shorter fields (Company / Date / Total) require a full phrase match
    return len(fragment_words) > 0 and match_count == len(fragment_words)


# ==========================================
# 3. DATA LOADING
# ==========================================
def load_sroie_split_hybrid(split_name):
    split_dir = os.path.join(DATASET_DIR, split_name)
    img_dir   = os.path.join(split_dir, "img")
    box_dir   = os.path.join(split_dir, "box")
    ent_dir   = os.path.join(split_dir, "entities")

    examples = {"image_path": [], "words": [], "bboxes": [], "ner_tags": []}

    for filename in os.listdir(img_dir):
        if not filename.endswith(".jpg"):
            continue

        file_id  = filename.split(".")[0]
        img_path = os.path.join(img_dir, filename)
        box_path = os.path.join(box_dir, file_id + ".txt")
        ent_path = os.path.join(ent_dir, file_id + ".txt")

        if not (os.path.exists(box_path) and os.path.exists(ent_path)):
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
            with open(ent_path, 'r', encoding='utf-8', errors='ignore') as f:
                entities = json.load(f)
        except Exception:
            continue

        doc_words, doc_bboxes, doc_ner_tags = [], [], []

        # --- Step 1: Read all lines ---
        raw_lines = []
        with open(box_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue

                coords        = [int(p) for p in parts[:8]]
                text_fragment = ",".join(parts[8:]).strip()

                x_min = min(coords[0], coords[2], coords[4], coords[6])
                y_min = min(coords[1], coords[3], coords[5], coords[7])
                x_max = max(coords[0], coords[2], coords[4], coords[6])
                y_max = max(coords[1], coords[3], coords[5], coords[7])

                if x_max <= x_min or y_max <= y_min:
                    continue

                raw_lines.append({
                    "text":  text_fragment,
                    "box":   [x_min, y_min, x_max, y_max],
                    "y_min": y_min
                })

        # --- Step 2: Sort top-to-bottom by Y-axis ---
        raw_lines = sorted(raw_lines, key=lambda x: x["y_min"])

        # --- Step 3: State-machine label assignment ---
        found_date  = False
        found_total = False

        for line_data in raw_lines:
            text_fragment = line_data["text"]
            box           = line_data["box"]
            norm_box      = normalize_bbox(box, width, height)

            assigned_label = "O"
            for key, full_string in entities.items():
                is_addr = (key.upper() == "ADDRESS")
                if fuzzy_match(text_fragment, full_string, is_address=is_addr):
                    assigned_label = key.upper()
                    break

            # Update state checkpoints
            if assigned_label == "TOTAL":
                found_total = True
            if assigned_label == "DATE":
                found_date = True

            # Lockouts: suppress late-appearing duplicates
            if assigned_label == "ADDRESS" and found_total:
                assigned_label = "O"
            if assigned_label == "COMPANY" and (found_date or found_total):
                assigned_label = "O"

            # Split fragment into individual words; all share the same bounding box
            for word in text_fragment.split():
                doc_words.append(word)
                doc_bboxes.append(norm_box)
                doc_ner_tags.append(LABEL2ID[assigned_label])

        examples["image_path"].append(img_path)
        examples["words"].append(doc_words)
        examples["bboxes"].append(doc_bboxes)
        examples["ner_tags"].append(doc_ner_tags)

    return Dataset.from_dict(examples)


# ==========================================
# 4. PREPROCESSING
# ==========================================
def prepare_dataset(batch):
    """Opens images on the fly and encodes the batch."""
    images = [Image.open(path).convert("RGB") for path in batch["image_path"]]

    processor = AutoProcessor.from_pretrained(MODEL_ID, apply_ocr=False)

    encoding = processor(
        images,
        batch["words"],
        boxes=batch["bboxes"],
        word_labels=batch["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    return encoding


# ==========================================
# 5. METRICS
# ==========================================
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Flatten and filter out -100 padding tokens
    true_predictions = [
        ID2LABEL[pred]
        for prediction, label in zip(predictions, labels)
        for pred, lab in zip(prediction, label)
        if lab != -100
    ]
    true_labels = [
        ID2LABEL[lab]
        for prediction, label in zip(predictions, labels)
        for pred, lab in zip(prediction, label)
        if lab != -100
    ]

    precision, recall, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average="weighted", zero_division=0
    )
    f1_macro  = f1_score(true_labels, true_predictions, average="macro",  zero_division=0)
    f1_micro  = f1_score(true_labels, true_predictions, average="micro",  zero_division=0)
    accuracy  = accuracy_score(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1_weighted,   # Used by metric_for_best_model="f1"
        "f1_macro":  f1_macro,
        "f1_micro":  f1_micro,
        "accuracy":  accuracy,
    }


# ==========================================
# 6. MAIN — MODEL INIT & TRAINING
# ==========================================
def main():
    print("Loading and parsing raw data with Hybrid logic...")
    raw_datasets = DatasetDict({
        "train": load_sroie_split_hybrid("train"),
        "test":  load_sroie_split_hybrid("test"),
    })


    encoded_datasets = raw_datasets.map(
        prepare_dataset,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # Re-initialize with a flat 5-label classification head
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        num_labels=len(LABELS),
        ignore_mismatched_sizes=True  # Drops the base model's default 2-label head
    )

    # ~3 logs/evals per epoch (79 steps per epoch)
    step_interval = math.ceil(79 / 3)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,

        # Optimizer
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,

        # Scheduler
        lr_scheduler_type="linear",
        learning_rate=2e-5,
        warmup_steps=50,

        # Logging & evaluation
        eval_strategy="steps",
        eval_steps=step_interval,
        save_strategy="steps",
        save_steps=step_interval,
        logging_steps=step_interval,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["test"],
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()