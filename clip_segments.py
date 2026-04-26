import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
CLIP_DIR = os.path.join(DATA_DIR, "clips")

os.makedirs(CLIP_DIR, exist_ok=True)

manifest_path = os.path.join(DATA_DIR, "segments_manifest.csv")
df = pd.read_csv(manifest_path)

print(f"Loaded {len(df)} segments")

updated_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    rid = row["recording_id"]
    seg_idx = row["segment_idx"]
    audio_path = row["audio_path"]
    start = float(row["start"])
    end = float(row["end"])

    clip_filename = f"{rid}_{seg_idx}.wav"
    clip_path = os.path.join(CLIP_DIR, clip_filename)

    if os.path.exists(clip_path):
        updated_rows.append(row)
        continue

    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        clip = audio[int(start*sr): int(end*sr)]

        if len(clip) == 0:
            continue

        sf.write(clip_path, clip, sr)

        row["audio_path"] = clip_path
        updated_rows.append(row)

    except Exception as e:
        print(f"Error clipping {rid}_{seg_idx}: {str(e)[:80]}")

new_manifest = pd.DataFrame(updated_rows)
new_manifest_path = os.path.join(DATA_DIR, "segments_clipped_manifest.csv")
new_manifest.to_csv(new_manifest_path, index=False)

print(f"\nClipping completed.")
print(f"Clipped dataset size: {len(new_manifest)}")
print(f"Saved to: {new_manifest_path}")