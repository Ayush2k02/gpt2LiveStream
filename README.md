# gpt2LiveStream

CPU-only GPT-2 inference in C++20. Loads Hugging Face `gpt2` weights from plain text dumps, runs the full transformer stack (layer norm, multi-head causal attention, MLP/GELU, tied LM head), and samples with top-k.

## Layout

| Path | Purpose |
|------|---------|
| `inference/` | C++ inference (`main.cc`) |
| `modelLoader/` | Export HF GPT-2 weights + tokenizer to `weights/` |
| `weights/` | Parameter `.txt` dumps + GPT-2 tokenizer (weight dumps are gitignored) |

## Prerequisites

- g++ with C++20 support
- Python 3 + `transformers` / `torch` (only to export weights)

## 1. Export weights

```bash
cd modelLoader
python -m venv .venv
source .venv/bin/activate
pip install torch transformers
python main.py
```

This writes `../weights/transformer.*.txt` and saves the tokenizer under `weights/tokenizer/`.

## 2. Build & run

```bash
cd inference
g++ -std=c++20 -O3 -o gpt2 main.cc -I.
./gpt2
```

On start, vocab and weights load once. Then you get a loop:

1. **prompt** — generation prompt (empty line quits)
2. **num tokens** — how many new tokens to sample

Defaults: temperature `0.8`, top-k `40`.

## Notes

- Weight files under `weights/*.txt` are large (~1.4GB) and ignored by git; regenerate them locally with `modelLoader/main.py`.
- Paths in the C++ loader are relative to `inference/` (`../weights/...`).
