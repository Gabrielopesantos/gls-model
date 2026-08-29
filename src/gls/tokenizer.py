"""Project tokenizer: train a byte-level BPE on the target corpus, and prove the
choice with a measured bake-off against a spread of off-the-shelf tokenizers.

Phase 0 fixes vocab at 32000 (128-aligned) and one tokenizer for all three
tiers - a decision with Phase 4 consequences (speculative decoding needs draft
and target to share a vocab), documented in
``privatedocs/plan/model-sizes.md``.

Subcommands::

    python -m gls.tokenizer fetch      # cache a fixed FineWeb-Edu slice locally
    python -m gls.tokenizer train      # train artifacts/tokenizer/tokenizer.json
    python -m gls.tokenizer compare    # leaderboard: chars/token, own vs refs
    python -m gls.tokenizer stats      # vocab coverage on the holdout
    python -m gls.tokenizer sweep      # chars/token vs training-corpus size
    python -m gls.tokenizer domains    # chars/token off-domain (code, other langs)

``compare`` decides which artifact ships: own must beat the best 32k reference
by >3% chars/token. A bigger-vocab tokenizer scoring higher on raw chars/token
is not an argument to adopt it - see the break-even section it prints.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

VOCAB_SIZE = 32_000

# Fixed slice sizes. The holdout is taken after skipping the whole training
# slice plus a gap, so the bake-off is measured on documents the own tokenizer
# never saw - the one correctness requirement of the exercise.
#
# 200k is well past diminishing returns, measured by `sweep` on the 10k holdout:
#   25k -> 4.608   50k -> 4.614 (+0.13%)   100k -> 4.616 (+0.05%)   200k -> 4.617 (+0.03%)
# The merge table is essentially converged by 25k docs; more corpus is wasted CPU.
TRAIN_DOCS = 200_000
HOLDOUT_GAP = 50_000
HOLDOUT_DOCS = 10_000

DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
TEXT_FIELD = "text"

# Bake-off references: two 32k, three ~50k, one ~150k. All ungated and carry a
# `tokenizer.json`. The ship decision only ever compares own against the best of
# the 32k rows - the larger ones are here to answer "would a bigger vocab have
# been better", which the break-even table in `compare` addresses head-on.
REFERENCES = [
    ("Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1"),
    ("Llama-2 (TinyLlama)", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("SmolLM2", "HuggingFaceTB/SmolLM2-360M"),
    ("GPT-2", "openai-community/gpt2"),
    ("GPT-NeoX-20B", "EleutherAI/gpt-neox-20b"),
    ("Qwen2.5", "Qwen/Qwen2.5-0.5B"),
]

# "KV tokens in remaining VRAM" per tier - from ../reference/hardware.md, a
# VRAM-budget number that is not part of the model config. d_model and
# kv_bytes/token come from gls.model.PRESETS via `_tiers()` so this file cannot
# drift from model-sizes.md.
_TIER_KV_TOKENS = {"small": 3_700_000, "medium": 447_000, "large": 190_000}


def _tiers() -> dict[str, dict]:
    """Per-tier figures for the `compare` break-even print, dimensions sourced
    from the model presets (single source of truth) and joined with the VRAM
    budget above."""
    from gls.model import PRESETS, kv_bytes_per_token

    return {
        name: {
            "d_model": PRESETS[name].d_model,
            "kv_bytes": kv_bytes_per_token(PRESETS[name]),
            "kv_tokens": kv_tokens,
        }
        for name, kv_tokens in _TIER_KV_TOKENS.items()
    }


# Reserved now, not bolted on in Phase 1 - adding special tokens later resizes
# the embedding with randomly initialised rows. endoftext + pad + a chat pair +
# a block of reserved slots, all counted toward VOCAB_SIZE by the trainer.
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    *[f"<|reserved_{i}|>" for i in range(12)],
]

# Ship own only if it beats the best 32k reference by more than this.
WIN_THRESHOLD = 0.03

_COL = 24  # first-column width in the compare/stats tables

# Corpus sizes for `sweep`, doubling up to the shipped TRAIN_DOCS.
SWEEP_POINTS = (25_000, 50_000, 100_000, 200_000)

# Off-domain probe for `domains`: (label, hf_name, config, text_field). All
# ungated and streamable. The point is to show, reproducibly, where an
# English-prose tokenizer degrades - soft on code/Romance/math, a wall on
# non-Latin scripts. Scope reasoning in privatedocs/plan/phase-0-baseline-model.md.
DOMAIN_SAMPLE = 1_500
DOMAIN_SETS = [
    ("Alpaca (En instruct)", "tatsu-lab/alpaca", None, "text"),
    ("Python (CodeSearchNet)", "code-search-net/code_search_net", "python", "func_code_string"),
    ("code, mixed (Rosetta)", "christopher/rosetta-code", None, "code"),
    ("math/LaTeX (OpenR1)", "open-r1/OpenR1-Math-220k", None, "problem"),
    ("Portuguese (wiki)", "wikimedia/wikipedia", "20231101.pt", "text"),
    ("German (wiki)", "wikimedia/wikipedia", "20231101.de", "text"),
    ("Arabic (wiki)", "wikimedia/wikipedia", "20231101.ar", "text"),
    ("Hindi (wiki)", "wikimedia/wikipedia", "20231101.hi", "text"),
    ("Japanese (wiki)", "wikimedia/wikipedia", "20231101.ja", "text"),
    ("Chinese (wiki)", "wikimedia/wikipedia", "20231101.zh", "text"),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_path() -> Path:
    return _repo_root() / "artifacts" / "tokenizer" / "tokenizer.json"


def _corpus_paths() -> dict[str, Path]:
    base = _repo_root() / "data" / "tokenizer-corpus"
    return {"train": base / "train.jsonl.gz", "holdout": base / "holdout.jsonl.gz"}


# --------------------------------------------------------------------------- #
# corpus                                                                      #
# --------------------------------------------------------------------------- #


def fetch(force: bool = False) -> None:
    """Stream the fixed FineWeb-Edu slice once and cache it as gzipped JSONL.

    Done as its own step so a long BPE train reads from local disk rather than
    holding a network stream open, and so ``compare`` reuses the exact same
    holdout deterministically.
    """
    from datasets import load_dataset

    paths = _corpus_paths()
    if not force and paths["train"].exists() and paths["holdout"].exists():
        print(f"corpus already cached at {paths['train'].parent} (use --force to refetch)")
        return

    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(DATASET, name=DATASET_CONFIG, split="train", streaming=True)
    it = iter(stream)

    def drain(dest: Path, count: int, label: str) -> None:
        with gzip.open(dest, "wt", encoding="utf-8") as fh:
            for i in range(count):
                text = next(it)[TEXT_FIELD]
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                if (i + 1) % 25_000 == 0:
                    print(f"  {label}: {i + 1}/{count}")
        print(f"wrote {dest}")

    drain(paths["train"], TRAIN_DOCS, "train")
    for _ in range(HOLDOUT_GAP):
        next(it)
    drain(paths["holdout"], HOLDOUT_DOCS, "holdout")

    # The streaming parquet reader can fault during interpreter shutdown; close
    # the generator deterministically now that we are done with it.
    it.close()


def _iter_texts(path: Path, limit: int | None = None) -> Iterator[str]:
    if not path.exists():
        raise SystemExit(f"missing {path} - run `python -m gls.tokenizer fetch` first")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in itertools.islice(fh, limit):
            yield json.loads(line)["text"]


# --------------------------------------------------------------------------- #
# train                                                                       #
# --------------------------------------------------------------------------- #


def _train_bpe(texts: Iterator[str], length: int, out: Path):
    """Train one byte-level BPE and save it, returning it. Shared by `train` and `sweep`."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        # All 256 byte symbols in the base alphabet => every byte string is
        # representable => no UNK, exact round-trip on arbitrary input.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=length >= TRAIN_DOCS,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return tok


def train() -> None:
    if not _corpus_paths()["train"].exists():
        fetch()

    out = _artifact_path()
    tok = _train_bpe(_iter_texts(_corpus_paths()["train"]), TRAIN_DOCS, out)

    got = tok.get_vocab_size()
    print(f"saved {out}  vocab={got}")
    if got != VOCAB_SIZE:
        raise SystemExit(f"vocab {got} != {VOCAB_SIZE}: corpus too small for the merge budget")
    if got % 128 != 0:
        raise SystemExit(f"vocab {got} is not 128-aligned")


# --------------------------------------------------------------------------- #
# measurement                                                                 #
# --------------------------------------------------------------------------- #


def _load_own():
    from tokenizers import Tokenizer

    path = _artifact_path()
    if not path.exists():
        raise SystemExit(f"missing {path} - run `python -m gls.tokenizer train` first")
    return Tokenizer.from_file(str(path))


def _load_named(repo: str):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))


class _Measure:
    __slots__ = ("vocab", "docs", "chars", "u_bytes", "tokens", "used")

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab
        self.docs = self.chars = self.u_bytes = self.tokens = 0
        self.used: Counter[int] = Counter()

    def add(self, text: str, ids: list[int]) -> None:
        self.docs += 1
        self.chars += len(text)
        self.u_bytes += len(text.encode("utf-8"))
        self.tokens += len(ids)
        self.used.update(ids)

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.tokens

    @property
    def bytes_per_token(self) -> float:
        return self.u_bytes / self.tokens


def _measure(tok, texts: Iterator[str]) -> _Measure:
    m = _Measure(tok.get_vocab_size())
    for text in texts:
        m.add(text, tok.encode(text, add_special_tokens=False).ids)
    return m


def _measurements() -> list[tuple[str, _Measure]]:
    """Own + every reference, all scored on the same held-out documents."""
    holdout = list(_iter_texts(_corpus_paths()["holdout"]))
    out = [("own", _measure(_load_own(), iter(holdout)))]
    for label, repo in REFERENCES:
        out.append((label, _measure(_load_named(repo), iter(holdout))))
    return out


# --------------------------------------------------------------------------- #
# compare/stats/sweep                                                     #
# --------------------------------------------------------------------------- #


def _best_32k(ms: list[tuple[str, _Measure]]) -> tuple[str, _Measure]:
    cands = [(label, m) for label, m in ms if label != "own" and m.vocab == VOCAB_SIZE]
    return max(cands, key=lambda pair: pair[1].chars_per_token)


def _print_breakeven(own: _Measure, ms: list[tuple[str, _Measure]]) -> None:
    print("\nis a bigger vocab worth it? extra bf16 embedding weights, against")
    print("the KV cache that better compression frees at the same context:")
    for tier, spec in _tiers().items():
        budget_chars = spec["kv_tokens"] * own.chars_per_token
        print(
            f"\n  {tier}: d_model {spec['d_model']}, KV {spec['kv_bytes'] // 1024} KiB/token, "
            f"~{spec['kv_tokens'] / 1000:.0f}k-token (~{budget_chars / 1e6:.1f}M-char) budget"
        )
        for label, m in ms:
            if m.vocab <= VOCAB_SIZE:
                continue
            extra_gib = (m.vocab - VOCAB_SIZE) * spec["d_model"] * 2 / 1024**3
            if m.chars_per_token <= own.chars_per_token:
                print(
                    f"    {label:<20} +{extra_gib:.3f} GiB, and compresses worse -> strictly worse"
                )
                continue
            saved_per_char = (1 / own.chars_per_token - 1 / m.chars_per_token) * spec["kv_bytes"]
            be = (m.vocab - VOCAB_SIZE) * spec["d_model"] * 2 / saved_per_char / 1e6
            budget = budget_chars / 1e6
            if be > budget:
                verdict = "never pays within VRAM"
            elif be > 0.6 * budget:
                verdict = "wash"
            else:
                verdict = "pays for long context"
            print(f"    {label:<20} +{extra_gib:.3f} GiB, break-even {be:.2f}M chars -> {verdict}")


def compare() -> None:
    ms = _measurements()
    own = ms[0][1]
    print(f"holdout: {own.docs} docs, {own.chars:,} chars\n")
    print(f"{'':<{_COL}}{'vocab':>10}{'chars/tok':>11}{'bytes/tok':>11}{'tokens':>14}")
    for label, m in ms:
        print(
            f"{label:<{_COL}}{m.vocab:>10,}{m.chars_per_token:>11.3f}{m.bytes_per_token:>11.3f}{m.tokens:>14,}"
        )

    ref_label, ref = _best_32k(ms)
    gain = own.chars_per_token / ref.chars_per_token - 1
    print(
        f"\nown vs best 32k ({ref_label}): {gain:+.1%} chars/token (threshold +{WIN_THRESHOLD:.0%})"
    )
    print("-> ship OWN tokenizer" if gain > WIN_THRESHOLD else "-> ship the 32k REFERENCE")

    _print_breakeven(own, ms)


def stats() -> None:
    ms = _measurements()
    print(f"{'':<{_COL}}{'vocab':>10}{'used':>10}{'unused':>10}{'unused %':>10}")
    for label, m in ms:
        used = len(m.used)
        unused = m.vocab - used
        print(f"{label:<{_COL}}{m.vocab:>10,}{used:>10,}{unused:>10,}{unused / m.vocab:>10.1%}")
    print("\nunused % is sample-size dependent (own reads higher on fewer docs) -")
    print("compare rows within one run, not across runs. A high count on a general")
    print("tokenizer is vocab spent on languages/code this corpus never uses.")


def sweep() -> None:
    """chars/token as a function of training-corpus size, holdout fixed.

    Turns `TRAIN_DOCS = 200_000` from an assertion into a curve. Trains to a
    temp dir - never touches the shipped artifact.
    """
    holdout = list(_iter_texts(_corpus_paths()["holdout"]))
    train_path = _corpus_paths()["train"]
    print(f"holdout: {len(holdout)} docs\n")
    print(f"{'docs':>10}{'chars/tok':>12}{'vs prev':>10}")

    prev: float | None = None
    with tempfile.TemporaryDirectory() as td:
        for n in SWEEP_POINTS:
            tok = _train_bpe(_iter_texts(train_path, limit=n), n, Path(td) / f"bpe_{n}.json")
            cpt = _measure(tok, iter(holdout)).chars_per_token
            delta = "-" if prev is None else f"{cpt / prev - 1:+.2%}"
            print(f"{n:>10,}{cpt:>12.3f}{delta:>10}")
            prev = cpt


def _domain_docs(hf_name: str, config: str | None, field: str) -> list[str]:
    from datasets import load_dataset

    kw = {"split": "train", "streaming": True}
    if config:
        kw["name"] = config
    ds = load_dataset(hf_name, **kw)
    docs = [row[field] for row in itertools.islice(ds, DOMAIN_SAMPLE * 2)]
    return [d for d in docs if d and len(d) > 50][:DOMAIN_SAMPLE]


def domains() -> None:
    """chars/token across domains the tokenizer was NOT trained for.

    Reproduces the off-domain table in phase-0-baseline-model.md: soft degrade
    on code/Romance/math, a hard wall (<1.2 chars/token) on non-Latin scripts.
    Network-heavy; a domain that fails to load is skipped, not fatal.
    """
    probes = [
        ("own", _load_own()),
        ("Mistral 32k", _load_named("mistralai/Mistral-7B-v0.1")),
        ("Qwen2.5 152k", _load_named("Qwen/Qwen2.5-0.5B")),
    ]
    rows = [
        ("FineWeb-Edu (baseline)", list(_iter_texts(_corpus_paths()["holdout"], DOMAIN_SAMPLE)))
    ]
    for label, hf_name, config, field in DOMAIN_SETS:
        try:
            rows.append((label, _domain_docs(hf_name, config, field)))
        except Exception as exc:  # noqa: BLE001 - a flaky/gated dataset must not abort the probe
            print(f"  skip {label}: {type(exc).__name__}: {str(exc)[:70]}")

    print(f"\n{'domain':<24}" + "".join(f"{name:>14}" for name, _ in probes) + "   chars/token")
    for label, docs in rows:
        chars = sum(len(d) for d in docs)
        cells = "".join(f"{chars / _tokens(tok, docs):>14.2f}" for _, tok in probes)
        print(f"{label:<24}{cells}   [{len(docs)} docs]")
    print("\nown beats the 32k reference on every row (byte-level whitespace) but")
    print("non-Latin scripts fall below ~1.1 - a 1024-token context there holds a")
    print("sentence or two. English-prose-only is a locked scope, not a preference.")


def _tokens(tok, docs: list[str]) -> int:
    return sum(len(tok.encode(d, add_special_tokens=False).ids) for d in docs)


# --------------------------------------------------------------------------- #
# cli                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gls.tokenizer", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="cache the fixed FineWeb-Edu slice locally")
    f.add_argument("--force", action="store_true", help="refetch even if cached")
    sub.add_parser("train", help="train the byte-level BPE artifact")
    sub.add_parser("compare", help="leaderboard: chars/token, own vs references")
    sub.add_parser("stats", help="vocab coverage on the holdout")
    sub.add_parser("sweep", help="chars/token vs training-corpus size")
    sub.add_parser("domains", help="chars/token across off-domain text (code, other langs)")

    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        fetch(force=args.force)
    elif args.cmd == "train":
        train()
    elif args.cmd == "compare":
        compare()
    elif args.cmd == "stats":
        stats()
    elif args.cmd == "sweep":
        sweep()
    elif args.cmd == "domains":
        domains()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
