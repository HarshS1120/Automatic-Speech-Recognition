import os
import pandas as pd
import torch
import librosa
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# ============================
# CONFIG
# ============================

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

manifest_path = os.path.join(DATA_DIR, "segments_clipped_manifest.csv")

print("Loading clipped dataset...")
df = pd.read_csv(manifest_path)

print(f"Loaded {len(df)} clipped segments")

# Keep only what we need
df = df[["audio_path", "text"]]

dataset = Dataset.from_pandas(df)

# ============================
# LOAD PROCESSOR + MODEL
# ============================

print("Loading Whisper processor + model...")

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="hi",
    task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Using device: {device}")

# ============================
# PREPARE DATA (NO Audio CAST)
# ============================

def prepare_batch(batch):
    # Load audio manually
    audio, sr = librosa.load(batch["audio_path"], sr=16000)

    # Extract features
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    batch["input_features"] = inputs.input_features[0]

    # Tokenize text
    labels = processor.tokenizer(
        batch["text"],
        return_tensors="pt"
    ).input_ids[0]

    batch["labels"] = labels

    return batch


print("Preparing dataset...")

dataset = dataset.map(
    prepare_batch,
    remove_columns=dataset.column_names,
    desc="Feature extraction"
)

print("Dataset ready for training.")

# ============================
# DATA COLLATOR
# ============================

data_collator = DataCollatorForSeq2Seq(
    processor.tokenizer,
    model=model,
    padding=True
)

# ============================
# TRAINING CONFIG
# ============================

import torch

def data_collator(features):
    # Convert input_features to tensors
    input_features = [
        torch.tensor(f["input_features"], dtype=torch.float32)
        for f in features
    ]

    batch = {}
    batch["input_features"] = torch.stack(input_features)

    # Pad labels properly
    label_features = [f["labels"] for f in features]

    labels_batch = processor.tokenizer.pad(
        {"input_ids": label_features},
        padding=True,
        return_tensors="pt",
    )

    labels = labels_batch["input_ids"]

    # Replace padding token id with -100 for loss masking
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels

    return batch

training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "whisper-small-hi"),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    predict_with_generate=False,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ============================
# TRAIN
# ============================

print("Starting training...")
trainer.train()

trainer.save_model()

print("Training complete. Model saved.")