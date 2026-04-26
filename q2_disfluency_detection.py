import os
import pandas as pd
import re

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

SEGMENT_MANIFEST = os.path.join(DATA_DIR, "segments_clipped_manifest.csv")
META_FILE = os.path.join(BASE_DIR, "ft_data.csv")

OUTPUT_FILE = os.path.join("outputs", "disfluency_dataset.csv")

os.makedirs("outputs", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(SEGMENT_MANIFEST)

print(f"Total segments: {len(df)}")

# ======================
# Load original metadata (contains URLs)
# ======================

meta_df = pd.read_csv(META_FILE)

# Ensure correct column names exist
if "rec_url_fixed" in meta_df.columns:
    url_column = "rec_url_fixed"
elif "rec_url_gcp" in meta_df.columns:
    url_column = "rec_url_gcp"
else:
    raise ValueError("No audio URL column found in metadata file.")

# Create mapping: recording_id → URL
id_to_url = dict(zip(meta_df["recording_id"], meta_df[url_column]))

# ======================
# Disfluency detection logic
# ======================

FILLERS = ["उम", "उम्म", "अह", "मतलब", "हम्म"]

def detect_repetition(text):
    words = text.split()
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            return True, f"repetition_{words[i]}"
    return False, None

def detect_prolongation(text):
    if re.search(r"(.)\1{2,}", text):
        return True, "prolongation"
    return False, None

def detect_fillers(text):
    for filler in FILLERS:
        if filler in text:
            return True, f"filler_{filler}"
    return False, None

def detect_disfluency(text):
    for func in [detect_fillers, detect_repetition, detect_prolongation]:
        detected, label = func(text)
        if detected:
            return True, label
    return False, None

# ======================
# Process segments
# ======================

rows = []

for _, row in df.iterrows():
    text = str(row["text"]).strip()
    recording_id = row["recording_id"]
    segment_idx = row["segment_idx"]

    detected, label = detect_disfluency(text)

    if not detected:
        continue

    rows.append({
        "recording_id": recording_id,
        "segment_idx": segment_idx,
        "disfluency_type": label,
        "audio_url": id_to_url.get(recording_id, ""),
        "text": text
    })

# ======================
# Save CSV
# ======================

result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDetected {len(result_df)} disfluency segments")
print(f"Saved to {OUTPUT_FILE}")