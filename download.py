import tiktoken
import torch
from datasets import load_dataset

# Parameters
seq_len = 512
n_samples = 2000  # How many docs to keep for quick dev
out_path = "wikitext2-tiktoken_512_seq_len.pt"

# 1. Make the tokenizer
enc = tiktoken.get_encoding("gpt2")
pad_id = enc.eot_token  # 50256

# 2. Download a dataset (change as desired)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
print(f"Loaded {len(ds)} documents from WikiText-2")

# 3. Tokenize and pad
def pad(ids, length, pad_id=50256):
    if len(ids) < length:
        return ids + [pad_id] * (length - len(ids))
    else:
        return ids[:length]

samples = []
for i, d in enumerate(ds):
    tokens = enc.encode(d["text"])
    tokens = pad(tokens, seq_len, pad_id=pad_id)
    samples.append(tokens)
    if i + 1 >= n_samples:
        break
samples = torch.tensor(samples, dtype=torch.long)  # [n_samples, seq_len]

# 4. Save as a .pt file
torch.save(samples, out_path)
print(f"Saved {samples.shape} tensor to {out_path}")
