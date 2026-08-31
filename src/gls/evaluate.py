"""Measurement for a fine-tune.

* ``perplexity`` - exp of the mean response-token cross-entropy on a held-out
  instruction split. Run it against the pretrained checkpoint and again against
  the fine-tuned one; the drop is the headline number.
* ``harness`` - score a checkpoint on ``lm-eval`` tasks (``lambada_openai``,
  ``arc_easy``, ``piqa``) through a tiny ``lm_eval.api.model.LM`` adapter over
  ``GLSModel.forward``. Teacher-forced sequence scoring only - no decode loop, no
  KV cache. Runs on our own forward pass, so no HF port and no Triton.
* ``export_hf`` - write a ``transformers``-loadable directory via the Llama
  rename converter (``gls.model.to_llama_state_dict``). Kept as the greedy-decode
  correctness reference and as a cross-check: the same tasks scored through the
  HF port should match the adapter's numbers within task noise.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from gls import checkpoint, paths, sft
from gls.model import llama_config_dict, to_llama_state_dict
from gls.trainer import Runtime

DEFAULT_TASKS = ("lambada_openai", "arc_easy", "piqa")


def _note(msg: str) -> None:
    print(f"[eval] {msg}", file=sys.stderr)


def _run_meta(ckpt_dir: Path) -> dict[str, Any]:
    """``train_config.json`` from the run that produced this checkpoint. Pass the
    resolved checkpoint dir (``runs/<name>/checkpoints/step-*``); the config
    sits two levels up. Lets ``ppl`` reproduce the exact holdout - same seed,
    same fraction, same block_size - without re-typing it.
    """
    for cand in (ckpt_dir / "train_config.json", ckpt_dir.parent.parent / "train_config.json"):
        if cand.exists():
            return json.loads(cand.read_text())
    return {}


def perplexity(
    ckpt: str | Path,
    corpus: str = "dolly",
    *,
    split: str = "val",
    batch_size: int = 16,
    iters: int = 200,
    seed: int | None = None,
    val_fraction: float | None = None,
    device: str | None = None,
) -> dict[str, float]:
    """Token-weighted mean response-token loss and its exp, over ``iters``
    sampled batches. ``seed``/``val_fraction`` default to the values recorded
    in the checkpoint's run config so the holdout matches training."""
    ckpt_dir = checkpoint.resolve_init(str(ckpt))
    meta = _run_meta(ckpt_dir)
    if not meta:
        _note(
            f"no train_config.json near {ckpt_dir}; holdout defaults to seed 1337, "
            f"val_fraction {sft.DEFAULT_VAL_FRACTION}, block_size = model max_seq_len"
        )
    the_seed = int(seed if seed is not None else meta.get("seed", 1337))
    the_vf = float(
        val_fraction
        if val_fraction is not None
        else (meta.get("val_fraction") or sft.DEFAULT_VAL_FRACTION)
    )

    rt = Runtime.resolve(_normalise_device(device))
    model = checkpoint.load_model_dir(ckpt_dir, rt.device)
    model.eval()

    # block_size must match training's - it decides which examples are dropped as
    # too long and which contexts are trimmed, so a mismatch carves a different
    # holdout. Training uses `cfg.block_size or max_seq_len` (gls.train._make_data).
    block_size = int(meta.get("block_size") or model.cfg.max_seq_len)
    src = sft.SFTData(corpus, block_size, val_fraction=the_vf, seed=the_seed)
    is_val = split == "val"
    gen = torch.Generator().manual_seed(the_seed + 7)

    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for _ in range(iters):
            x, y = src.batch(
                batch_size, rt.device, val=is_val, generator=gen, pin_memory=rt.pin_memory
            )
            with rt.autocast:
                logits, _ = model(x)
            total_loss += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                y.reshape(-1),
                ignore_index=sft.IGNORE_INDEX,
                reduction="sum",
            ).item()
            total_tok += int((y != sft.IGNORE_INDEX).sum().item())

    mean = total_loss / max(total_tok, 1)
    out = {"loss": mean, "ppl": math.exp(mean), "tokens": float(total_tok)}
    _note(f"{ckpt_dir}  {split}  loss {mean:.4f}  ppl {out['ppl']:.2f}  ({total_tok} tokens)")
    return out


def export_hf(ckpt: str | Path, out_dir: str | Path) -> Path:
    """Write ``config.json`` + ``model.safetensors`` + tokenizer files that
    ``transformers.AutoModelForCausalLM``/``lm-eval`` can load directly."""
    try:
        from transformers import (  # pyright: ignore[reportMissingImports]
            LlamaConfig,
            LlamaForCausalLM,
            PreTrainedTokenizerFast,
        )
    except ImportError:
        raise SystemExit("transformers not installed - run `uv sync --group eval`") from None

    ckpt_dir = checkpoint.resolve_init(str(ckpt))
    model = checkpoint.load_model_dir(ckpt_dir)

    hf = LlamaForCausalLM(LlamaConfig(**llama_config_dict(model.cfg)))
    missing, unexpected = hf.load_state_dict(to_llama_state_dict(model), strict=False)
    if unexpected or set(missing) - {"lm_head.weight"}:
        raise SystemExit(f"converter mismatch: missing={missing} unexpected={unexpected}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(out)
    PreTrainedTokenizerFast(
        tokenizer_file=str(paths.tokenizer_artifact()),
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    ).save_pretrained(out)
    _note(f"exported {ckpt_dir} -> {out}")
    return out


def _normalise_device(device: str | None) -> str:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return "cuda" if dev == "gpu" else dev


def _make_adapter(ckpt: str | Path, device: str, batch_size: int):
    """A minimal ``lm_eval.api.model.LM`` over ``GLSModel`` - teacher-forced
    sequence scoring only, no decode loop. Built here so importing ``gls.evaluate``
    never imports ``lm_eval``."""
    from lm_eval.api.model import LM  # pyright: ignore[reportMissingImports]

    from gls import data

    rt = Runtime.resolve(device)
    model = checkpoint.load_model_dir(checkpoint.resolve_init(str(ckpt)), rt.device)
    model.eval()
    tok = data.load_tokenizer()
    eot = data._eot_id(tok)
    max_len = model.cfg.max_seq_len

    def _score(ctx_ids: list[int], cont_ids: list[int]) -> tuple[float, bool]:
        full = ctx_ids + cont_ids
        # forward consumes full[:-1]; keep it within the context window, trimming
        # from the left (oldest context first), same as lm-eval's HF model.
        if len(full) - 1 > max_len:
            overflow = len(full) - 1 - max_len
            full = full[overflow:]
            ctx_len = max(1, len(ctx_ids) - overflow)
        else:
            ctx_len = max(1, len(ctx_ids))
        inp = torch.tensor([full[:-1]], device=rt.device)
        with torch.no_grad(), rt.autocast:
            logits, _ = model(inp)
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        tgt = torch.tensor(full[ctx_len:], device=rt.device)
        rows = logp[ctx_len - 1 : ctx_len - 1 + len(tgt)]
        tok_logp = rows.gather(-1, tgt[:, None]).squeeze(-1)
        is_greedy = bool((rows.argmax(-1) == tgt).all())
        return float(tok_logp.sum()), is_greedy

    class _GLSAdapter(LM):
        def __init__(self) -> None:
            super().__init__()
            self.batch_size = batch_size

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            from tqdm import tqdm

            out = []
            for req in tqdm(requests, disable=disable_tqdm):
                context, continuation = req.arguments
                ctx_ids = tok.encode(context, add_special_tokens=False).ids or [eot]
                cont_ids = tok.encode(continuation, add_special_tokens=False).ids
                out.append(_score(ctx_ids, cont_ids))
            return out

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            # A correct rolling score windows the document across max_seq_len and
            # sums the pieces; _score would just left-truncate to the tail. None
            # of the chosen tasks (lambada/arc/piqa) need this - a wikitext-style
            # task would. Refuse rather than return a silently wrong number.
            raise NotImplementedError(
                "gls has no windowed rolling scorer; the scoring tasks do not need one"
            )

        def generate_until(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError(
                "this adapter does teacher-forced scoring only; generation-style tasks are "
                "not wired through it (the decode loop lives in gls.inference)"
            )

    return _GLSAdapter()


def harness(
    ckpt: str | Path,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    *,
    limit: int | None = None,
    batch_size: int = 1,
    device: str | None = None,
) -> dict[str, Any]:
    """Score a checkpoint on ``lm-eval`` tasks through our own ``GLSModel``
    forward pass (no HF port, no decode loop). ``limit`` caps docs per task (a
    fast smoke; leave unset for real scores)."""
    try:
        from lm_eval import simple_evaluate  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise SystemExit("lm-eval not installed - run `uv sync --group eval`") from None

    dev = _normalise_device(device)
    lm = _make_adapter(ckpt, dev, batch_size)
    kwargs: dict[str, Any] = {"model": lm, "tasks": list(tasks), "limit": limit}
    results: Any = simple_evaluate(**kwargs)
    scored = (results or {}).get("results", {})
    table = {t: scored[t] for t in tasks if t in scored}
    for name, scores in table.items():
        _note(f"{name}: {scores}")
    return table
