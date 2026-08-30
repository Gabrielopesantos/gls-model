"""The ``gls`` command line.

All argparse construction lives here. The library modules expose plain functions;
this is the only place a CLI-shaped string becomes a domain call, and the only place
that hard-exits the interpreter.
"""

from __future__ import annotations

import argparse
import os
import sys

from gls import checkpoint, data, env, sft, tokenizer
from gls.config import add_dataclass_args, resolve
from gls.model import PRESETS
from gls.train import TrainConfig, train


def _cmd_train(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gls train", description="run a pretraining loop")
    p.add_argument("--config", default=None, help="TOML run config; CLI flags override it")
    add_dataclass_args(p, TrainConfig)
    cfg = resolve(TrainConfig, p, argv)
    if cfg.tier not in PRESETS:  # validated here so nothing downstream has to
        raise SystemExit(f"unknown tier {cfg.tier!r}; choose from {list(PRESETS)}")
    if cfg.init_from:
        # resolve now so a bad path fails at the boundary, before any work. If
        # --resume also applies (an interrupted fine-tune re-run with the same
        # config), resume wins in train() and this seed is ignored - not a
        # conflict, so no error here.
        checkpoint.resolve_init(cfg.init_from)
    train(cfg)
    return 0


def _cmd_data(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gls data")
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare", help="encode a corpus split to packed uint16 shards")
    prep.add_argument("--corpus", default="tinystories", choices=list(data.CORPORA))
    prep.add_argument("--split", default="train", choices=["train", "val"])
    prep.add_argument("--limit", type=int, default=None, help="cap document count (dev)")
    prep.add_argument("--shard-tokens", type=int, default=data.DEFAULT_SHARD_TOKENS)
    prep.add_argument("--force", action="store_true", help="rebuild even if present")
    a = p.parse_args(argv)
    data.prepare(a.corpus, a.split, limit=a.limit, force=a.force, shard_tokens=a.shard_tokens)
    return 0


_TOKENIZER_CMDS = {
    "train": "train the byte-level BPE artifact",
    "compare": "leaderboard: chars/token, own vs references",
    "stats": "vocab coverage on the holdout",
    "sweep": "chars/token vs training-corpus size",
    "domains": "chars/token across off-domain text (code, other langs)",
}


def _cmd_tokenizer(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gls tokenizer")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="cache the fixed FineWeb-Edu slice locally")
    f.add_argument("--force", action="store_true", help="refetch even if cached")
    for name, help_txt in _TOKENIZER_CMDS.items():
        sub.add_parser(name, help=help_txt)
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        tokenizer.fetch(force=a.force)
    else:
        getattr(tokenizer, a.cmd)()
    return 0


def _cmd_eval(argv: list[str]) -> int:
    from gls import evaluate

    p = argparse.ArgumentParser(prog="gls eval", description="before/after fine-tuning measurement")
    sub = p.add_subparsers(dest="cmd", required=True)

    ppl = sub.add_parser("ppl", help="response-token perplexity on a held-out instruction split")
    ppl.add_argument("--ckpt", required=True, help="checkpoint dir or run dir")
    ppl.add_argument("--corpus", default="dolly", choices=list(sft.SFT_CORPORA))
    ppl.add_argument("--split", default="val", choices=["train", "val"])
    ppl.add_argument("--iters", type=int, default=200)
    ppl.add_argument("--batch-size", type=int, default=16)
    ppl.add_argument("--seed", type=int, default=None, help="default: the run's recorded seed")
    ppl.add_argument("--device", default=None)

    exp = sub.add_parser("export", help="write a transformers-loadable dir via the Llama converter")
    exp.add_argument("--ckpt", required=True)
    exp.add_argument("--out", required=True)

    har = sub.add_parser("harness", help="score a checkpoint on lm-eval tasks via GLSModel")
    har.add_argument("--ckpt", required=True, help="checkpoint dir or run dir")
    har.add_argument("--tasks", default=",".join(evaluate.DEFAULT_TASKS))
    har.add_argument("--limit", type=int, default=None, help="cap docs/task (smoke run)")
    har.add_argument("--device", default=None, help="e.g. cpu, cuda, gpu; default auto")

    a = p.parse_args(argv)
    if a.cmd == "ppl":
        evaluate.perplexity(
            a.ckpt,
            a.corpus,
            split=a.split,
            iters=a.iters,
            batch_size=a.batch_size,
            seed=a.seed,
            device=a.device,
        )
    elif a.cmd == "export":
        evaluate.export_hf(a.ckpt, a.out)
    else:
        evaluate.harness(a.ckpt, tuple(a.tasks.split(",")), limit=a.limit, device=a.device)
    return 0


def _cmd_model(argv: list[str]) -> int:
    from gls import viz

    p = argparse.ArgumentParser(prog="gls model", description="inspect a tier's architecture")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="per-layer table + param / activation / KV sizes")
    s.add_argument("--tier", default="medium", choices=list(PRESETS))
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--block-size", type=int, default=None, help="default: tier max_seq_len")
    s.add_argument(
        "--graph", default=None, help="also write a module diagram here (needs graphviz)"
    )
    a = p.parse_args(argv)
    viz.summary(a.tier, batch_size=a.batch_size, block_size=a.block_size)
    if a.graph:
        viz.graph(a.tier, a.graph, block_size=a.block_size)
    return 0


def _cmd_env(argv: list[str]) -> int:
    argparse.ArgumentParser(prog="gls env", description="print torch/CUDA/device facts").parse_args(
        argv
    )
    info = env.describe()
    for key in ("python", "torch", "torch_cuda", "cuda_available", "device_count"):
        print(f"{key}: {info[key]}")
    for d in info["devices"]:
        print(
            f"  [{d['index']}] {d['name']} sm_{d['capability'].replace('.', '')} "
            f"{d['total_memory_gib']} GiB {d['multi_processor_count']} SMs"
        )
    if not info["cuda_available"]:
        print("cuda unavailable, CPU only", file=sys.stderr)
        return 1
    return 0


_COMMANDS = {
    "train": _cmd_train,
    "data": _cmd_data,
    "tokenizer": _cmd_tokenizer,
    "eval": _cmd_eval,
    "model": _cmd_model,
    "env": _cmd_env,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="gls", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("command", choices=list(_COMMANDS))
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="arguments for the subcommand (try `gls train -h`)"
    )
    ns = parser.parse_args(argv)
    return _COMMANDS[ns.command](ns.args)


def run() -> None:
    """Console-script entry point (``[project.scripts] gls``).

    ``--stream`` and the corpus preparers keep an HF parquet reader alive on a
    background thread that can fault during interpreter shutdown. Flush and
    hard-exit so a cosmetic teardown crash cannot mask a clean run.
    """
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    run()
