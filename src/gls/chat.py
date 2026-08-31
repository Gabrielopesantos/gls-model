"""Terminal chat with a checkpoint: text in, text out.

This is the top of the inference stack. It owns the ChatML framing (the same
``<|im_start|>``/``<|im_end|>`` strings SFT supervised, imported from ``gls.sft``),
a streaming detokenizer that reassembles UTF-8 as tokens arrive, and the REPL.
Everything below - the KV cache, the sampler - is ``gls.inference`` and
``gls.sampling``.

Two modes:

* **chat** (default) - wrap each turn in ChatML and stop on ``<|im_end|>``. Aimed
  at a fine-tuned checkpoint (``runs/small-dolly-sft``). Multi-turn works (the KV
  cache is kept across turns) but is outside the single-turn SFT distribution, so
  the second turn prints a note.
* **raw** (``--raw``) - feed the prompt verbatim, stop on ``<|endoftext|>``. For a
  base checkpoint, whose ChatML embedding rows are still zero.

The model's words go to stdout; everything else - the ``you ›``/``gls ›``
speaker labels, notes, diagnostics - goes to stderr. So an interactive session
reads with clear turns, and ``gls chat --prompt ... > out.txt`` captures only the
completion. One-shot (``--prompt``) mode prints no labels at all.
"""

from __future__ import annotations

import codecs
import sys
from functools import lru_cache
from pathlib import Path

import torch

from gls.inference import ContextFull, Session
from gls.sampling import SamplingConfig
from gls.sft import ASSISTANT_OPEN, TURN_CLOSE, USER_OPEN

EOS = "<|endoftext|>"


def _note(msg: str) -> None:
    print(f"[chat] {msg}", file=sys.stderr)


# Speaker labels for the REPL. They go to stderr and the model's words to stdout,
# so the two are already separable when piped; the labels (and the dim styling on
# a TTY) are the affordance for a human reading along.
_YOU, _GLS = "you", "gls"


def _labels() -> tuple[str, str]:
    """``(you-prompt, gls-prompt)``, dimmed when stderr is a terminal."""
    dim, reset = ("\033[2m", "\033[0m") if sys.stderr.isatty() else ("", "")
    return f"{dim}{_YOU} ›{reset} ", f"{dim}{_GLS} ›{reset} "


# --------------------------------------------------------------------------- #
# streaming detokenizer                                                        #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's ``bytes_to_unicode``: the printable-unicode stand-in
    char back to its raw byte. ``tokenizers``' ``ByteLevel`` uses exactly this
    table, so ``id_to_token`` strings decode through it byte-for-byte."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = {b: b for b in printable}
    n = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = 256 + n
            n += 1
    return {chr(v): k for k, v in mapping.items()}


class StreamDecoder:
    """Token ids in, text out, one token at a time.

    A byte-level BPE token is a byte string, not a character string: a multi-byte
    character can straddle two tokens. An incremental UTF-8 decoder holds an
    incomplete trailing sequence until the next token completes it, and turns a
    genuinely invalid byte into U+FFFD so one bad token cannot wedge the stream.
    """

    def __init__(self, tok) -> None:
        self._tok = tok
        self._map = _byte_decoder()
        self._utf8 = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token_id: int) -> str:
        piece = self._tok.id_to_token(token_id)
        if piece is None:
            return ""
        try:
            raw = bytes(self._map[ch] for ch in piece)
        except KeyError:
            raw = piece.encode("utf-8")  # an added/special token, stored literally
        return self._utf8.decode(raw)

    def flush(self) -> str:
        """Whatever bytes remain, decoded with replacement. Call once at the end."""
        return self._utf8.decode(b"", final=True)


# --------------------------------------------------------------------------- #
# turn rendering + the loop                                                    #
# --------------------------------------------------------------------------- #


def _render_turn(message: str, *, first: bool, raw: bool) -> str:
    if raw:
        return message
    opener = USER_OPEN if first else f"{TURN_CLOSE}\n{USER_OPEN}"
    return f"{opener}{message}{ASSISTANT_OPEN}"


def _speak(
    session: Session,
    tok,
    decoder: StreamDecoder,
    cfg: SamplingConfig,
    generator: torch.Generator,
    *,
    max_new_tokens: int,
    stop_ids: frozenset[int],
    out,
) -> None:
    """Stream one response. Drops the stop token from the session so it is not
    context for - or a repetition-penalty target in - the next turn."""
    stopped = False
    for token in session.generate(cfg, generator, max_new_tokens=max_new_tokens, stop_ids=stop_ids):
        if token in stop_ids:
            stopped = True
            break
        out.write(decoder.push(token))
        out.flush()
    out.write(decoder.flush())
    out.flush()
    if stopped:
        session.discard_last()


def chat(
    ckpt: str | Path,
    *,
    prompt: str | None = None,
    raw: bool = False,
    max_new_tokens: int = 256,
    sampling: SamplingConfig | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> None:
    from gls import checkpoint, data
    from gls.trainer import Runtime

    cfg = sampling or SamplingConfig()
    rt = Runtime.resolve(device)
    ckpt_dir = checkpoint.resolve_init(str(ckpt))
    model = checkpoint.load_model_dir(ckpt_dir, rt.device)
    model.eval()
    tok = data.load_tokenizer()

    stop_names = (EOS,) if raw else (TURN_CLOSE, EOS)
    stop_ids = frozenset(i for i in (tok.token_to_id(n) for n in stop_names) if i is not None)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    session = Session(model, rt.device, autocast=rt.autocast)

    def turn(message: str, *, first: bool) -> None:
        ids = tok.encode(_render_turn(message, first=first, raw=raw), add_special_tokens=False).ids
        session.push(ids)
        try:
            _speak(
                session,
                tok,
                StreamDecoder(tok),
                cfg,
                generator,
                max_new_tokens=max_new_tokens,
                stop_ids=stop_ids,
                out=sys.stdout,
            )
        except ContextFull as full:
            _note(f"context full ({full.length}/{session.max_seq_len}) - /reset to start over")
        sys.stdout.write("\n")
        sys.stdout.flush()

    if prompt is not None:
        # One-shot: the completion, undecorated, to stdout.
        turn(prompt, first=True)
        return

    you_prompt, gls_prompt = _labels()
    _note(f"{ckpt_dir.name} on {rt.device} - message + Enter, /reset to clear, Ctrl-D to exit")
    n = 0
    while True:
        sys.stderr.write(f"\n{you_prompt}")
        sys.stderr.flush()
        line = sys.stdin.readline()
        if not line:  # EOF
            sys.stderr.write("\n")
            break
        message = line.strip()
        if not message:
            continue
        if message == "/reset":
            session.reset()
            n = 0
            _note("context cleared")
            continue
        n += 1
        if n == 2 and not raw:
            _note("turn 2+ is outside the single-turn SFT distribution - answers may drift")
        sys.stderr.write(gls_prompt)
        sys.stderr.flush()
        turn(message, first=(n == 1))
