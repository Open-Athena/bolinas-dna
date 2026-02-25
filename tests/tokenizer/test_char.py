"""Tests for the DNA character-level tokenizer with BOS/EOS (HuggingFace API)."""

import tempfile

import pytest
from transformers import AutoTokenizer

from bolinas.tokenizer.char import SPECIAL_TOKENS, create_char_tokenizer

# ---------------------------------------------------------------------------
# Vocab sizes
# ---------------------------------------------------------------------------


def test_vocab_size():
    tok = create_char_tokenizer()
    assert tok.vocab_size == 8


def test_special_token_ids():
    tok = create_char_tokenizer()
    assert tok.convert_tokens_to_ids("[PAD]") == 0
    assert tok.convert_tokens_to_ids("[UNK]") == 1
    assert tok.convert_tokens_to_ids("[BOS]") == 2
    assert tok.convert_tokens_to_ids("[EOS]") == 3


def test_base_ordering():
    tok = create_char_tokenizer()
    assert tok.convert_tokens_to_ids("a") == 4
    assert tok.convert_tokens_to_ids("c") == 5
    assert tok.convert_tokens_to_ids("g") == 6
    assert tok.convert_tokens_to_ids("t") == 7


# ---------------------------------------------------------------------------
# BOS / EOS token properties
# ---------------------------------------------------------------------------


def test_bos_eos_properties():
    tok = create_char_tokenizer()
    assert tok.bos_token == "[BOS]"
    assert tok.eos_token == "[EOS]"
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


def test_call_returns_batch_encoding():
    tok = create_char_tokenizer()
    out = tok("acgt")
    assert "input_ids" in out
    assert "attention_mask" in out
    # BOS + a c g t + EOS = 6 tokens
    assert out["input_ids"] == [2, 4, 5, 6, 7, 3]
    assert out["attention_mask"] == [1, 1, 1, 1, 1, 1]


def test_call_batch():
    tok = create_char_tokenizer()
    out = tok(["acgt", "ggaa"])
    assert len(out["input_ids"]) == 2
    # Each sequence: BOS + 4 bases + EOS = 6
    assert len(out["input_ids"][0]) == 6
    assert len(out["input_ids"][1]) == 6


# ---------------------------------------------------------------------------
# encode — with BOS/EOS (default)
# ---------------------------------------------------------------------------


def test_encode_single_base():
    tok = create_char_tokenizer()
    ids = tok.encode("a")
    assert ids == [2, 4, 3]  # BOS a EOS


def test_encode_sequence():
    tok = create_char_tokenizer()
    ids = tok.encode("acgt")
    assert ids == [2, 4, 5, 6, 7, 3]  # BOS a c g t EOS


def test_encode_case_insensitive():
    tok = create_char_tokenizer()
    lower = tok.encode("acgt")
    upper = tok.encode("ACGT")
    mixed = tok.encode("AcGt")
    assert lower == upper == mixed
    assert lower == [2, 4, 5, 6, 7, 3]


def test_encode_empty():
    tok = create_char_tokenizer()
    # Empty input still gets BOS + EOS
    assert tok.encode("") == [2, 3]


def test_encode_n_maps_to_unk():
    tok = create_char_tokenizer()
    ids = tok.encode("n")
    assert ids == [2, 1, 3]  # BOS [UNK] EOS


# ---------------------------------------------------------------------------
# encode — without special tokens
# ---------------------------------------------------------------------------


def test_encode_no_special_tokens():
    tok = create_char_tokenizer()
    ids = tok.encode("acgt", add_special_tokens=False)
    assert ids == [4, 5, 6, 7]


def test_encode_empty_no_special_tokens():
    tok = create_char_tokenizer()
    assert tok.encode("", add_special_tokens=False) == []


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def test_decode_roundtrip():
    tok = create_char_tokenizer()
    seq = "acgtttggg"
    decoded = tok.decode(tok.encode(seq), skip_special_tokens=True)
    assert decoded == seq


def test_decode_skip_special_tokens():
    tok = create_char_tokenizer()
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    a_id = tok.convert_tokens_to_ids("a")
    assert tok.decode([bos_id, a_id, eos_id], skip_special_tokens=True) == "a"


def test_decode_empty():
    tok = create_char_tokenizer()
    assert tok.decode([]) == ""


# ---------------------------------------------------------------------------
# Lookup: convert_tokens_to_ids / convert_ids_to_tokens
# ---------------------------------------------------------------------------


def test_convert_tokens_to_ids():
    tok = create_char_tokenizer()
    assert isinstance(tok.convert_tokens_to_ids("a"), int)


def test_convert_ids_to_tokens():
    tok = create_char_tokenizer()
    assert tok.convert_ids_to_tokens(4) == "a"


def test_convert_special_tokens():
    tok = create_char_tokenizer()
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.convert_tokens_to_ids(name) == i
        assert tok.convert_ids_to_tokens(i) == name


# ---------------------------------------------------------------------------
# Roundtrip / uniqueness
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip():
    tok = create_char_tokenizer()
    seq = "acgtacgtacgt"
    decoded = tok.decode(tok.encode(seq), skip_special_tokens=True)
    assert decoded == seq


def test_all_bases_unique_ids():
    tok = create_char_tokenizer()
    ids = {tok.convert_tokens_to_ids(b) for b in "acgt"}
    assert len(ids) == 4


# ---------------------------------------------------------------------------
# save_pretrained / AutoTokenizer.from_pretrained
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip():
    tok = create_char_tokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        tok.save_pretrained(tmpdir)
        loaded = AutoTokenizer.from_pretrained(tmpdir)

        seq = "acgtttggg"
        assert loaded.encode(seq) == tok.encode(seq)
        assert loaded.encode(seq) == [2, 4, 5, 6, 7, 7, 7, 6, 6, 6, 3]
        assert loaded.decode(loaded.encode(seq), skip_special_tokens=True) == seq
        assert loaded.bos_token == "[BOS]"
        assert loaded.eos_token == "[EOS]"
        assert loaded.bos_token_id == tok.bos_token_id
        assert loaded.eos_token_id == tok.eos_token_id
