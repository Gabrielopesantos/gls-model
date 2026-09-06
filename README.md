# gls-model

A decoder-only transformer built from scratch — tokenizer, data pipeline,
pretraining loop, instruction fine-tuning, evaluation, and KV-cached inference —
with no training framework in the path. The loop mechanics (forward/backward,
optimizer step, LR schedule, checkpoint/resume, distributed hooks) stay visible
in the source rather than living inside `Trainer` or `accelerate`.

The emphasis is infrastructure and compute: reproducible training on rented GPUs,
and (the deepest focus) a mini-vLLM-style paged-KV-cache serving path.

## Architecture

One `ModelConfig`, three named presets, no per-tier branches anywhere in the
model code — size is data, not code. Every preset is expressible as a
`transformers` `LlamaConfig`, so `gls eval export` produces a reference model
through a pure rename table (`gls.model.to_llama_state_dict`).

- Grouped-query attention (cuts the KV cache 4×)
- Rotary position embeddings, GPT-NeoX (`rotate_half`) layout — no learned
  position parameters
- RMSNorm, SwiGLU, tied input/output embeddings, no bias terms
- `F.scaled_dot_product_attention` for the attention kernel

| tier   | d_model | layers | heads | kv_heads | head_dim | d_ff | max_seq_len | params |
| ------ | ------- | ------ | ----- | -------- | -------- | ---- | ----------- | ------ |
| small  | 384     | 6      | 6     | 2        | 64       | 1024 | 512         | ~21.7M |
| medium | 1024    | 24     | 16    | 4        | 64       | 2816 | 2048        | ~303M  |
| large  | 2048    | 24     | 16    | 4        | 128      | 5504 | 2048        | ~1.13B |

Tokenizer: own byte-level BPE, vocab 32000, trained once on a FineWeb-Edu slice
and shared by every tier. Committed at `artifacts/tokenizer/tokenizer.json`.

## Findings

**`medium` (303M) pretrain — FineWeb-Edu `sample-10BT`, rented A100 40GB.**

| metric      | value                                              |
| ----------- | -------------------------------------------------- |
| tokens      | 5.24B (20000 steps × 262144)                        |
| wall clock  | ~23.5 h                                             |
| throughput  | 62.2k tok/s                                         |
| peak memory | 21.9 GiB of 39.5 GiB, flat across the run           |
| MFU         | ~42% (~131 TFLOP/s against the 312 TFLOP/s bf16 peak) |

17.3 tokens/param against Chinchilla's ~20 — the size/data split was about right;
what bounds the result is the compute budget (~9.5e18 FLOPs, ~1/9 of GPT-2 XL).
Output is well-formed and frequently wrong, which is what this scale buys.

**Instruction fine-tune — Dolly-15k, response-token loss on a 750-example holdout.**

| checkpoint | loss   | perplexity |
| ---------- | ------ | ---------- |
| base       | 2.8666 | 17.58      |
| after SFT  | 2.4174 | 11.22      |

−0.449 nats, −36% perplexity. On the lm-eval harness, `arc_easy` and `piqa` stay
flat (instruction tuning adds no knowledge) while `lambada_openai` improves ~4σ
(+3.65pp accuracy) — a domain-shift effect toward ordinary well-formed prose, not
new capability. SFT's job here was format: ChatML framing, turn termination,
answer shape.

Full write-up, including the five measurement defects the run exposed, in
`privatedocs/reference/medium-run-results.md`.

## Layout

```
src/gls/
  model       GLSModel, ModelConfig, PRESETS — architecture + forward, no optimizer
  trainer     Trainer, Runtime — the optimizer, the step, the eval pass
  train       TrainConfig, train, lr_at — the hand-written loop
  data        PackedData, StreamingTokens, TokenSource — sampling interface
  sft         instruction-tuning data (ChatML, response-only loss mask)
  tokenizer   byte-level BPE trainer + the reference bake-off
  checkpoint  atomic save/load, rotation, resume
  evaluate    response-token perplexity, lm-eval adapter, HF export
  sampling    SamplingConfig, sample — logits → one token id
  inference   Session — single-request generation over a KV cache
  chat        the terminal REPL + ChatML framing
  tracking    JSONL metrics + optional Weights & Biases mirror
  config      dataclass ⇄ argparse ⇄ TOML
  paths / env one root for every on-disk location; torch/CUDA facts
  cli         the `gls` command — argparse lives here and nowhere else
configs/      committed run configs (TOML)
infra/        rented-GPU runbook and scripts
privatedocs/  design notes, phase plans, run results (not tracked)
tests/        pytest — no GPU, no network; the tokenizer artifact is committed
```

The inference path (`model` + `checkpoint` + `sampling` + `inference`) never
imports the training stack; `tests/test_package.py` enforces that seam.

## Quickstart

The environment is [devenv](https://devenv.sh) + [uv](https://docs.astral.sh/uv/);
`direnv allow` (or `devenv shell`) brings up Python 3.12, torch on the CUDA 12.9
wheel index, and the `gls` console script.

```
gls env                                              # torch / CUDA / device facts
gls data prepare --corpus tinystories --split train --limit 120000
gls data prepare --corpus tinystories --split val   --limit 12000
gls train --config configs/small-tinystories.toml    # ~9 min on a consumer GPU

gls train --config configs/small-dolly-sft.toml      # instruction fine-tune
gls eval  ppl --ckpt runs/small-dolly-sft --corpus dolly
gls chat      --ckpt runs/small-dolly-sft
```

The tokenizer artifact is committed; `gls tokenizer {train,compare,stats,sweep,domains}`
retrains it and regenerates the bake-off numbers.

`gls <command> -h` for the full flag set. Commands: `train`, `data`, `tokenizer`,
`eval`, `model`, `chat`, `env`.

Run configs layer **defaults < `--config file.toml` < explicit flags**. The
`medium` configs (`configs/medium-*.toml`) are rental shapes — see `infra/` for
the rented-GPU flow.

## Status

| phase | area                                    | state                              |
| ----- | --------------------------------------- | ---------------------------------- |
| 0     | baseline model, tokenizer, training loop | done                               |
| 1     | instruction fine-tuning + evaluation    | done (`small` pipeline + `medium` before/after) |
| 1.5   | training infra                          | single-process done; DDP is the remaining scope |
| 2     | KV-cache inference + `gls chat`         | shipped; HF greedy-decode equivalence gate not yet built |
| 3     | paged KV-cache + continuous batching    | not started — the primary focus    |
| 4     | quantization, speculative decode, Triton, MoE | opportunistic                 |
