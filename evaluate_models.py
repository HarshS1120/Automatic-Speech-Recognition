import os
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from tqdm import tqdm
import pandas as pd

# ============================
# CONFIG
# ============================

BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "models", "whisper-small-hi")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================
# LOAD FLEURS (Hindi)
# ============================

print("Loading Hindi FLEURS test set...")

fleurs = load_dataset("google/fleurs", "hi_in", split="test")

print(f"Test samples: {len(fleurs)}")

# ============================
# LOAD PROCESSOR
# ============================

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="hi",
    task="transcribe"
)

# ============================
# LOAD MODELS
# ============================

print("Loading baseline model...")
baseline_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
).to(device)

print("Loading fine-tuned model...")
finetuned_model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR
).to(device)

baseline_model.eval()
finetuned_model.eval()

wer_metric = evaluate.load("wer")

# ============================
# EVALUATION FUNCTION
# ============================

def evaluate_model(model):
    predictions = []
    references = []

    for example in tqdm(fleurs):
        audio = example["audio"]["array"]

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                max_length=225
            )

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        predictions.append(transcription)
        references.append(example["transcription"])

    wer = wer_metric.compute(
        predictions=predictions,
        references=references
    )

    return wer, predictions, references

# ============================
# RUN EVALUATION
# ============================

print("\nEvaluating baseline model...")
baseline_wer, baseline_preds, refs = evaluate_model(baseline_model)

print("\nEvaluating fine-tuned model...")
finetuned_wer, finetuned_preds, _ = evaluate_model(finetuned_model)

# ============================
# SAVE RESULTS
# ============================

results_df = pd.DataFrame({
    "reference": refs,
    "baseline_prediction": baseline_preds,
    "finetuned_prediction": finetuned_preds
})

os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/predictions_comparison.csv", index=False)

wer_df = pd.DataFrame({
    "Model": ["Whisper-small (baseline)", "Whisper-small (fine-tuned)"],
    "WER": [baseline_wer, finetuned_wer]
})

wer_df.to_csv("outputs/wer_results.csv", index=False)

print("\n====================")
print("FINAL RESULTS")
print("====================")
print(wer_df)