import os
import pandas as pd
import unicodedata
import re

BASE_DIR = os.getcwd()
WORDS_FILE = os.path.join("data", "unique_words.csv")
OUTPUT_FILE = os.path.join("outputs", "spelling_analysis.csv")

os.makedirs("outputs", exist_ok=True)

print("Loading words...")
df = pd.read_csv(WORDS_FILE)
words = df.iloc[:, 0].astype(str).tolist()

print(f"Total unique words: {len(words)}")

# ======================
# Normalize
# ======================
def normalize(word):
    return unicodedata.normalize("NFC", word.strip())

words = [normalize(w) for w in words]

# ======================
# Basic Devanagari Validation
# ======================
def is_devanagari(word):
    return all("\u0900" <= ch <= "\u097F" for ch in word)

# ======================
# Heuristic Spelling Checks
# ======================
def looks_incorrect(word):

    # 1. Repeated character >=3 times
    if re.search(r"(.)\1{2,}", word):
        return True

    # 2. Single isolated matra (diacritic) at start
    if len(word) > 0 and "\u093e" <= word[0] <= "\u094d":
        return True

    # 3. Too short single character words
    if len(word) == 1:
        return True

    return False

clean_words = []
results = []

for w in words:
    if not is_devanagari(w):
        continue

    label = "incorrect spelling" if looks_incorrect(w) else "correct spelling"
    results.append({"word": w, "label": label})

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)

print("\n====== FINAL COUNT ======")
print(f"Correct words: {len(result_df[result_df.label=='correct spelling'])}")
print(f"Incorrect words: {len(result_df[result_df.label=='incorrect spelling'])}")
print(f"Saved to {OUTPUT_FILE}")