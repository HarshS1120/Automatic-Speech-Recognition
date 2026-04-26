# 🎙️ Hindi ASR Fine-Tuning with Whisper

This project implements an end-to-end Automatic Speech Recognition (ASR) pipeline by fine-tuning OpenAI's Whisper-small model on ~10 hours of Hindi conversational speech data.

It includes data preprocessing, model training, evaluation, disfluency detection, spelling validation, and a robust lattice-based evaluation strategy.

---

## Key Results

- **Model:** Whisper-small (Transformer-based seq2seq)
- **Dataset:** ~10 hours Hindi speech
- **Evaluation Dataset:** FLEURS Hindi (test split)
- **Metric:** Word Error Rate (WER)

| Model | WER |
|------|------|
| Pretrained Whisper-small | 0.718 |
| Fine-tuned Whisper-small | 0.404 |

✅ **43.7% relative WER improvement**

---

##  Methodology

### 1. Data Preprocessing
- Fixed broken cloud storage URLs
- Downloaded audio and transcription metadata
- Converted audio to **16 kHz mono**
- Segmented recordings using timestamp-aligned transcriptions
- Removed short/empty utterances
- Cleaned and normalized Hindi text

### 2. Feature Extraction
- Extracted **log-Mel spectrograms** using WhisperProcessor
- Tokenized Hindi text labels for seq2seq training

### 3. Model Training
- Fine-tuned **Whisper-small (multilingual)**
- Framework: PyTorch + HuggingFace Transformers
- Custom data collator for speech-text batching
- GPU training (Colab)

---

##  Evaluation

- Evaluated on **FLEURS Hindi test dataset**
- Metric: **Word Error Rate (WER)**

WER formula: (Substitutions + Deletions + Insertions) / Total Words


Fine-tuning significantly improved transcription accuracy by adapting the model to conversational Hindi speech.

---

##  Additional Components

### Disfluency Detection
- Rule-based detection of:
  - Fillers (e.g., "उम", "मतलब")
  - Repetitions (e.g., "मैं मैं")
  - Prolongations (e.g., "हम्म्म")
- Extracted timestamp-based audio clips for each disfluency

---

### Spelling Validation
- Processed **1.75L+ unique Devanagari words**
- Applied Unicode normalization and rule-based filtering
- Detected structural anomalies (invalid characters, repetitions, malformed tokens)

---

### Robust Evaluation (Lattice-Based)
- Constructed a **confusion network** from multiple ASR outputs
- Used **majority voting (≥60%)** to correct noisy references
- Improved fairness in WER computation

---

##  Tech Stack

- Python
- PyTorch
- HuggingFace Transformers & Datasets
- Librosa / Torchaudio
- Pydub
- Google Colab

---
