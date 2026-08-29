"""Tokenizer artifact checks.

Tests using the `tok` fixture skip when the artifact is absent, so the suite
stays green before `python -m gls.tokenizer train` has been run (mirrors
test_env.py). Constant-only tests always run.
"""

import pytest

from gls.tokenizer import (
    DOMAIN_SETS,
    REFERENCES,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    _artifact_path,
    _tiers,
)


@pytest.fixture(scope="module")
def tok():
    if not _artifact_path().exists():
        pytest.skip("tokenizer artifact not trained yet (`python -m gls.tokenizer train`)")
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(_artifact_path()))


# --- constants: no artifact, no network needed ---------------------------------


def test_references_well_formed():
    labels = [label for label, _ in REFERENCES]
    assert len(labels) == len(set(labels))
    assert all("/" in repo for _, repo in REFERENCES)


def test_tiers_derived_from_model_presets():
    from gls.model import PRESETS

    tiers = _tiers()
    assert set(tiers) == {"small", "medium", "large"}
    # model-sizes.md#tiers: KV/token is 24 KiB at medium, 48 KiB at large.
    assert tiers["medium"]["kv_bytes"] == 24 * 1024
    assert tiers["large"]["kv_bytes"] == 48 * 1024
    for name, spec in tiers.items():
        assert spec["d_model"] == PRESETS[name].d_model
        assert spec["d_model"] % 128 == 0


def test_domain_sets_well_formed():
    labels = [row[0] for row in DOMAIN_SETS]
    assert len(labels) == len(set(labels))
    for label, hf_name, config, field in DOMAIN_SETS:
        assert label and "/" in hf_name and field
        assert config is None or isinstance(config, str)


# Bytes that must survive a round-trip: ASCII, whitespace runs (indented code,
# tabs, newlines), CJK, emoji, and a lone combining mark.
ROUNDTRIP_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n\t\treturn x ** 2  # trailing spaces   \n\n\n",
    "     leading and    interior     spaces",
    "日本語のトークン化とハンズオンの深さ",
    "emoji: 🦣🧠🚀 and a flag 🇵🇹",
    "zoë - café - naïve - ́ combining",
]


@pytest.mark.parametrize("text", ROUNDTRIP_SAMPLES)
def test_exact_roundtrip(tok, text):
    assert tok.decode(tok.encode(text, add_special_tokens=False).ids) == text


def test_arbitrary_bytes_roundtrip_without_unk(tok):
    blob = bytes(range(256)).decode("latin-1") * 3
    ids = tok.encode(blob, add_special_tokens=False).ids
    assert tok.token_to_id("<unk>") is None
    assert tok.decode(ids) == blob


def test_vocab_size_exact_and_128_aligned(tok):
    assert tok.get_vocab_size() == VOCAB_SIZE
    assert VOCAB_SIZE % 128 == 0


def test_special_tokens_present_and_leading(tok):
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(name) == i


def test_special_tokens_not_split(tok):
    ids = tok.encode("<|endoftext|>", add_special_tokens=False).ids
    assert ids == [tok.token_to_id("<|endoftext|>")]
