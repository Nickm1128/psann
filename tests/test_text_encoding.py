"""Encoding hygiene recognizes byte-decoding mistakes, not a list of symbols."""

import json
from pathlib import Path

import pytest

from tools.text_encoding import find_mojibake

ROOT = Path(__file__).resolve().parents[1]


def corrupt(text, encoding):
    # Preserve undefined Windows-1252 bytes as their Latin-1 control characters,
    # as permissive Windows decoders do. The fixtures stay readable UTF-8 source.
    return "".join(
        (
            bytes([byte]).decode(encoding, errors="surrogateescape")
            if encoding == "latin-1" or byte not in {0x81, 0x8D, 0x8F, 0x90, 0x9D}
            else chr(byte)
        )
        for byte in text.encode("utf-8")
    )


@pytest.mark.parametrize("encoding", ["cp1252", "latin-1"])
@pytest.mark.parametrize(
    "symbol",
    [
        "→",
        "←",
        "↔",
        "⇒",
        "‘",
        "’",
        "“",
        "”",
        "–",
        "—",
        "…",
        "≈",
        "≤",
        "≥",
        "∞",
        "∑",
        "√",
        "∆",
        "×",
        "±",
        "÷",
        "π",
        "é",
        "中",
        "😀",
    ],
)
def test_corrupted_utf8_symbols_report_original_character_and_exact_offset(symbol, encoding):
    prefix = "café 中文: "
    damaged = corrupt(symbol, encoding)
    assert find_mojibake(prefix + damaged + " end") == [(len(prefix), damaged, symbol)]


@pytest.mark.parametrize(
    "text",
    [
        "café ≈ 2 — 中文",
        "→ ← ↔ ⇒ ‘quotes’ “quotes” – — …",
        "x ≤ y ≥ 0; ∞ ∑ √ ∆ × ± ÷ π",
        "Français: Noël, São Paulo; Â Ã â",
        "日本語 한국어 العربية Ελληνικά русский 😀",
        "ASCII -> <= >= - and escaped \\u2192",
    ],
)
def test_legitimate_unicode_and_ascii_remain_accepted(text):
    assert find_mojibake(text) == []


def test_multiple_corruptions_and_replacement_character_are_all_reported():
    first, second = corrupt("→", "cp1252"), corrupt("≈", "latin-1")
    text = first + " café " + second + "\ufffd"
    assert find_mojibake(text) == [
        (0, first, "→"),
        (len(first) + 6, second, "≈"),
        (len(text) - 1, "\ufffd", None),
    ]


@pytest.mark.parametrize(
    "invalid", [b"\xc0\xaf", b"\xe0\x80\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2\x86"]
)
def test_invalid_or_incomplete_utf8_is_not_reported_as_recoverable(invalid):
    assert find_mojibake(invalid.decode("latin-1")) == []


def test_canonical_notebook_zip_messages_and_decoded_cells_are_clean():
    notebook = json.loads(
        (ROOT / "notebooks/PSANN_Parity_and_Probes.ipynb").read_text(encoding="utf-8")
    )
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    messages = [
        line.strip()
        for source in sources
        for line in source.splitlines()
        if 'print(f"Zipped ' in line
    ]
    assert messages == ['print(f"Zipped {folder_path} → {output_path}")'] * 2
    assert [
        (index, find_mojibake(source))
        for index, source in enumerate(sources)
        if find_mojibake(source)
    ] == []
