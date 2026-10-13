# gpt2LiveStream

GPT-2 (124M) inference from scratch, with no ML libraries — just plain arrays and loops. Loads Hugging Face `gpt2` weights from text dumps, runs the full transformer stack (byte-pair tokenizer, layer norm, multi-head causal attention, MLP/GELU, tied LM head), and samples with top-k.

There are two implementations of the same model:

- **TypeScript** (`inference/main.ts`, runs on Bun) — the main one, with a **KV cache** so each new token only runs through the model once.
- **C++20** (`inference/main.cc`) — the original port; recomputes the full sequence every step.

## Layout

| Path | Purpose |
|------|---------|
| `inference/main.ts` | TypeScript inference (tokenizer, model, KV cache, sampling) |
| `inference/main.cc` | C++20 inference |
| `inference/include/json.hpp` | nlohmann/json, used by the C++ tokenizer loader |
| `modelLoader/` | Exports HF GPT-2 weights + tokenizer into `weights/` |
| `weights/` | Parameter `.txt` dumps (gitignored) + GPT-2 tokenizer (tracked) |

## Prerequisites

- [Bun](https://bun.sh) for the TypeScript version
- g++ or clang++ with C++20 support for the C++ version
- [uv](https://docs.astral.sh/uv/) (or Python 3.12+ with `torch` and `transformers`) — only to export weights

## 1. Export weights

The weight dumps (~1.4 GB) aren't in the repo, so generate them once:

```bash
cd modelLoader
uv run main.py
```

This writes `../weights/transformer.*.txt`, one file per parameter, and saves the tokenizer under `weights/tokenizer/`.

## 2. Run

### TypeScript

```bash
cd inference
bun install
bun start
```

### C++

```bash
cd inference
g++ -std=c++20 -O3 -o gpt2 main.cc -I.
./gpt2
```

Both load the vocab and weights once, then loop:

1. **prompt** — text to continue (empty line quits)
2. **num tokens** — how many new tokens to generate

Output streams token by token. Sampling defaults: temperature `0.8`, top-k `40`. Generation stops early at `<|endoftext|>` or the 1024-token context limit.

## Notes

- Run both programs from `inference/` — weight and tokenizer paths are relative (`../weights/...`).
- The KV cache preallocates about 75 MB per generation (12 layers × K/V × 1024 positions × 768 floats).
- `bun run typecheck` type-checks the TypeScript.
