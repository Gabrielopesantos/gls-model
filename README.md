# gls-model

A from-scratch decoder-only transformer in PyTorch, built to learn the full LLM training pipeline.

## What's here so far

- Byte-level BPE tokenizer (vocab 32,000), trained on a FineWeb-Edu slice.
- Decoder-only model: grouped-query attention, RoPE, RMSNorm, SwiGLU, tied embeddings. Tiers: small (~21.7M), medium (~303M), large (~1.13B).
- Pretraining, including the ~303M config on FineWeb-Edu (`configs/medium-fineweb.toml`). TinyStories is the small smoke corpus.
- Supervised fine-tuning on Dolly-15k (`configs/medium-dolly-sft.toml`).
- Inference and generation over a KV cache (`gls chat`).
- Checkpoints and resume (`--resume auto`): safetensors weights plus optimizer and RNG state.
- Evaluation: held-out response perplexity, and lm-eval (`lambada_openai`, `arc_easy`, `piqa`) through the model's own forward pass.
- GPU provisioning scripts in `infra/vast/`.

## Usage

Install with `uv sync`. The entry point is `gls`.

```
gls tokenizer train
gls data prepare --corpus fineweb-edu --split train
gls train --config configs/medium-fineweb.toml
gls train --config configs/medium-fineweb.toml --resume auto
gls train --config configs/medium-dolly-sft.toml
gls eval ppl --ckpt runs/medium-dolly-sft --corpus dolly
gls eval harness --ckpt runs/medium-dolly-sft
gls chat --ckpt runs/medium-dolly-sft --prompt "Hello"
```

Smaller end-to-end check: `configs/small-tinystories.toml`, then `configs/small-dolly-sft.toml`.
