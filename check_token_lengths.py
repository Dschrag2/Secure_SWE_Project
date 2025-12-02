import torch
from datasets import load_dataset
from transformers import RobertaTokenizer
import numpy as np

print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# --- Check BigVul Dataset ---
print("\n" + "="*60)
print("ANALYZING BIGVUL DATASET")
print("="*60)

print("Loading BigVul dataset...")
bigvul_dataset = load_dataset("bstee615/bigvul", "default", split="train")

print("Tokenizing samples (this may take a moment)...")
def get_token_length(example):
    tokens = tokenizer(example["func_before"], truncation=False, add_special_tokens=True)
    return {"token_length": len(tokens["input_ids"])}

# Sample a subset for speed (check 5000 samples)
sample_size = min(5000, len(bigvul_dataset))
bigvul_sample = bigvul_dataset.select(range(sample_size))
bigvul_with_lengths = bigvul_sample.map(get_token_length)

lengths = bigvul_with_lengths["token_length"]
print(f"\nBigVul Token Length Statistics (from {sample_size} samples):")
print(f"  Mean:    {np.mean(lengths):.2f} tokens")
print(f"  Median:  {np.median(lengths):.2f} tokens")
print(f"  Min:     {np.min(lengths)} tokens")
print(f"  Max:     {np.max(lengths)} tokens")
print(f"  95th percentile: {np.percentile(lengths, 95):.2f} tokens")
print(f"  99th percentile: {np.percentile(lengths, 99):.2f} tokens")

# Show percentage that fit in different limits
print(f"\nPercentage of samples that fit:")
for limit in [128, 256, 384, 512]:
    pct = np.sum(np.array(lengths) <= limit) / len(lengths) * 100
    print(f"  <= {limit} tokens: {pct:.1f}%")

# --- Check Juliet Dataset ---
print("\n" + "="*60)
print("ANALYZING JULIET DATASET")
print("="*60)

print("Loading Juliet dataset...")
juliet_dataset = load_dataset("LorenzH/juliet_test_suite_c_1_3", "default", split="train")

# Flatten the dataset first
def restructure_function(examples):
    new_codes = []
    goods = examples['good']
    bads = examples['bad']

    for good_code, bad_code in zip(goods, bads):
        if good_code:
            new_codes.append(good_code)
        if bad_code:
            new_codes.append(bad_code)

    return {"code": new_codes}

juliet_flat = juliet_dataset.map(restructure_function, batched=True, remove_columns=juliet_dataset.column_names)

print("Tokenizing samples...")
def get_token_length_juliet(example):
    tokens = tokenizer(example["code"], truncation=False, add_special_tokens=True)
    return {"token_length": len(tokens["input_ids"])}

# Sample a subset for speed
sample_size_juliet = min(5000, len(juliet_flat))
juliet_sample = juliet_flat.select(range(sample_size_juliet))
juliet_with_lengths = juliet_sample.map(get_token_length_juliet)

lengths_juliet = juliet_with_lengths["token_length"]
print(f"\nJuliet Token Length Statistics (from {sample_size_juliet} samples):")
print(f"  Mean:    {np.mean(lengths_juliet):.2f} tokens")
print(f"  Median:  {np.median(lengths_juliet):.2f} tokens")
print(f"  Min:     {np.min(lengths_juliet)} tokens")
print(f"  Max:     {np.max(lengths_juliet)} tokens")
print(f"  95th percentile: {np.percentile(lengths_juliet, 95):.2f} tokens")
print(f"  99th percentile: {np.percentile(lengths_juliet, 99):.2f} tokens")

print(f"\nPercentage of samples that fit:")
for limit in [128, 256, 384, 512]:
    pct = np.sum(np.array(lengths_juliet) <= limit) / len(lengths_juliet) * 100
    print(f"  <= {limit} tokens: {pct:.1f}%")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("Based on the statistics above, choose a max_length that captures")
print("at least 90-95% of your samples without excessive truncation.")
