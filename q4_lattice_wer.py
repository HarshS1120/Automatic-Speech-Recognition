import jiwer
from collections import Counter
import numpy as np

# =====================================
# SIMPLE ALIGNMENT (Position-based)
# =====================================

def align_sequences(sequences):
    """
    Very simple alignment: pad shorter sequences.
    For real research you’d use dynamic alignment.
    """

    token_lists = [s.split() for s in sequences]
    max_len = max(len(t) for t in token_lists)

    aligned = []

    for t in token_lists:
        padded = t + ["<eps>"] * (max_len - len(t))
        aligned.append(padded)

    return aligned


# =====================================
# BUILD CONFUSION NETWORK
# =====================================

def build_lattice(model_outputs):
    aligned = align_sequences(model_outputs)
    max_len = len(aligned[0])

    lattice = []

    for i in range(max_len):
        position_tokens = [aligned[m][i] for m in range(len(aligned))]
        token_counts = Counter(position_tokens)
        lattice.append(token_counts)

    return lattice


# =====================================
# CONSENSUS DECISION
# =====================================

def derive_consensus(lattice, trust_threshold=0.6):
    consensus_tokens = []

    total_models = sum(lattice[0].values())

    for position in lattice:
        token, count = position.most_common(1)[0]

        if count / total_models >= trust_threshold:
            consensus_tokens.append(token)
        else:
            consensus_tokens.append(token)

    return " ".join([t for t in consensus_tokens if t != "<eps>"])


# =====================================
# MODIFIED WER
# =====================================

def lattice_based_wer(model_output, consensus_transcript):
    return jiwer.wer(consensus_transcript, model_output)


# =====================================
# FULL PIPELINE
# =====================================

def evaluate_models(models_outputs, reference):
    lattice = build_lattice(models_outputs)

    consensus = derive_consensus(lattice)

    print("\nOriginal Reference:")
    print(reference)
    print("\nConsensus Transcript:")
    print(consensus)

    print("\nWER Scores (Original Reference):")
    for i, model in enumerate(models_outputs):
        print(f"Model {i+1}: {jiwer.wer(reference, model)}")

    print("\nWER Scores (Lattice Consensus):")
    for i, model in enumerate(models_outputs):
        print(f"Model {i+1}: {lattice_based_wer(model, consensus)}")


# =====================================
# EXAMPLE USAGE
# =====================================

if __name__ == "__main__":

    reference = "मुझे लगता है कि भारत एक महान देश है"

    model_outputs = [
        "मुझे लगता है कि भारत एक महान देश है",
        "मुझे लगता है कि भारत एक महान देश है",
        "मुझे लगता है कि भारत एक महान देस है",
        "मुझे लगता है कि भारत एक महान देश है",
        "मुझे लगता है कि भारत एक महन देश है",
    ]

    evaluate_models(model_outputs, reference)