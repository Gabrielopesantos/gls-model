"""The ``gls`` command line.

All argparse construction lives here. The library modules expose plain functions;
this is the only place a CLI-shaped string becomes a domain call, and the only place
that hard-exits the interpreter.
"""

from __future__ import annotations

import argparse
import os
import sys

from gls import data, env, tokenizer
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
