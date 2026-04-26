import os
import requests
import pandas as pd
import json
from tqdm import tqdm

# =============================
# FOLDER SETUP
# =============================

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "raw_audio")
TRANS_DIR = os.path.join(DATA_DIR, "transcriptions")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANS_DIR, exist_ok=True)

print("Folders ready.")

# =============================
# LOAD METADATA CSV
# =============================

csv_path = "ft_data.csv"   # <-- put CSV in root folder
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} recordings")


# =============================
# URL FIXING FUNCTION
# =============================

def fix_url(old_url, is_transcription=False):
    if pd.isna(old_url) or not isinstance(old_url, str):
        return old_url

    if not old_url.startswith("https://"):
        old_url = "https://" + old_url.lstrip("/")

    if "joshtalks-data-collection" in old_url or "hq_data/hi" in old_url:
        if "/hi/" in old_url:
            suffix = old_url.split("/hi/")[-1]
        else:
            parts = old_url.split("/")
            suffix = "/".join(parts[-2:])

        if is_transcription:
            suffix = suffix.replace("_audio.wav", "_transcription.json")

        bucket = "upload_goai"

        return f"https://storage.googleapis.com/{bucket}/{suffix}"

    return old_url


df["audio_url"] = df["rec_url_gcp"].apply(fix_url)
df["trans_url"] = df["transcription_url_gcp"].apply(
    lambda x: fix_url(x, is_transcription=True)
)

# =============================
# DOWNLOAD FUNCTION
# =============================

def download_file(url, save_path):
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"Download failed: {url[:70]} -> {str(e)[:60]}")
        return False


# =============================
# DOWNLOAD AUDIO + TRANSCRIPTS
# =============================

DOWNLOAD_LIMIT = 200  # increase later

for _, row in tqdm(df.head(DOWNLOAD_LIMIT).iterrows(),
                   total=min(DOWNLOAD_LIMIT, len(df))):

    rid = str(row["recording_id"])

    audio_path = os.path.join(AUDIO_DIR, f"{rid}.wav")
    trans_path = os.path.join(TRANS_DIR, f"{rid}.json")

    # Download audio
    if not os.path.exists(audio_path):
        download_file(row["audio_url"], audio_path)

    # Download transcription
    if not os.path.exists(trans_path):
        download_file(row["trans_url"], trans_path)

print("Download completed.")

# =============================
# BUILD SEGMENT MANIFEST
# =============================

records = []

for _, row in tqdm(df.head(DOWNLOAD_LIMIT).iterrows(),
                   total=min(DOWNLOAD_LIMIT, len(df)),
                   desc="Building segments"):

    rid = str(row["recording_id"])
    audio_path = os.path.join(AUDIO_DIR, f"{rid}.wav")
    trans_path = os.path.join(TRANS_DIR, f"{rid}.json")

    if not os.path.exists(audio_path) or not os.path.exists(trans_path):
        continue

    try:
        with open(trans_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "segments" in data:
            segments = data["segments"]
        elif isinstance(data, list):
            segments = data
        else:
            continue

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                continue

            start = seg.get("start", 0)
            end = seg.get("end", row.get("duration", 0))

            records.append({
                "recording_id": rid,
                "segment_idx": i,
                "audio_path": audio_path,
                "text": text,
                "start": start,
                "end": end,
                "duration": end - start
            })

    except Exception as e:
        print(f"Transcription parse error {rid}: {str(e)[:50]}")

# Save manifest
manifest_df = pd.DataFrame(records)
manifest_path = os.path.join(DATA_DIR, "segments_manifest.csv")
manifest_df.to_csv(manifest_path, index=False)

print(f"\nCreated {len(manifest_df)} segments")
print(f"Manifest saved at: {manifest_path}")