"""gls.chat - ChatML turn rendering and the streaming detokenizer.

The `tok` tests skip without the tokenizer artifact (mirrors test_sft.py).
"""

from __future__ import annotations

import pytest

from gls import sft
from gls.chat import EOS, StreamDecoder, _labels, _render_turn
from gls.paths import tokenizer_artifact


@pytest.fixture(scope="module")
def tok():
    if not tokenizer_artifact().exists():
        pytest.skip("tokenizer artifact not trained yet (`gls tokenizer train`)")
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_artifact()))


def test_first_turn_prompt_matches_sft_prompt_span(tok):
    """The tokens a live turn feeds the model must be exactly the prompt span
    SFT supervised for the same instruction (no context)."""
    msg = "Give me three uses for a paperclip."
    ex = {"instruction": msg, "context": "", "response": "x"}
    enc = sft.encode(ex, sft.SFT_CORPORA["dolly"], tok, block_size=512)
    assert enc is not None
    ids, prompt_len, _ = enc

    rendered = _render_turn(msg, first=True, raw=False)
    assert tok.encode(rendered, add_special_tokens=False).ids == ids[:prompt_len]


def test_second_turn_reopens_the_previous_turn(tok):
    rendered = _render_turn("and again", first=False, raw=False)
    assert rendered.startswith(sft.TURN_CLOSE)
    assert rendered.endswith(sft.ASSISTANT_OPEN)


def test_raw_mode_is_verbatim():
    assert _render_turn("Once upon a time", first=True, raw=True) == "Once upon a time"


def test_speaker_labels_are_plain_when_stderr_is_not_a_tty(capsys):
    you, gls = _labels()  # pytest captures stderr, so isatty() is False here
    assert (you, gls) == ("you › ", "gls › ")
    assert "\033" not in you + gls


def test_stream_decoder_reassembles_multibyte(tok):
    """A byte-level BPE splits multi-byte characters across tokens; the decoder
    must buffer the partial sequence and emit it once complete."""
    text = "café — 日本語 🎉 ok"
    ids = tok.encode(text, add_special_tokens=False).ids
    dec = StreamDecoder(tok)
    out = "".join(dec.push(i) for i in ids) + dec.flush()
    assert out == text


def test_stream_decoder_never_emits_partial_utf8(tok):
    ids = tok.encode("日本語", add_special_tokens=False).ids
    dec = StreamDecoder(tok)
    for i in ids[:-1]:
        piece = dec.push(i)
        piece.encode("utf-8")  # whatever came out is always valid text
    dec.push(ids[-1])


def test_eos_is_a_real_token(tok):
    assert tok.token_to_id(EOS) is not None
