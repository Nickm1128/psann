"""Detect likely UTF-8 decoded as Windows-1252/Latin-1 in maintained text.

This development helper reports evidence for review; it never rewrites text.
An otherwise valid sequence can be intentional prose, so findings are suspicious
rather than proof of corruption. Single accented characters are not findings.
"""

from __future__ import annotations

# Accept either C1 controls from Latin-1 or their Windows-1252 punctuation
# equivalents, including decoders that preserve undefined Windows bytes as C1.
_BYTE_VALUES = {chr(byte): byte for byte in range(256)}
for _byte in range(0x80, 0xA0):
    try:
        _BYTE_VALUES[bytes([_byte]).decode("cp1252")] = _byte
    except UnicodeDecodeError:
        pass


def find_mojibake(text: str) -> list[tuple[int, str, str | None]]:
    """Return (character offset, observed text, recovered character) findings.

    Recognize every strictly valid 2-, 3-, or 4-byte UTF-8 sequence reconstructed
    from the two legacy decodings. This covers symbols beyond known examples.
    Reject overlong encodings, surrogates, and out-of-range code points through
    Python's strict UTF-8 decoder. A replacement character is irrecoverable and
    is reported with ``None`` as its recovered value.
    """
    findings: list[tuple[int, str, str | None]] = []
    values = [_BYTE_VALUES.get(character, -1) for character in text]
    position = 0
    while position < len(text):
        if text[position] == "\ufffd":
            findings.append((position, text[position], None))
        lead = values[position]
        width = (
            2
            if 0xC2 <= lead <= 0xDF
            else 3 if 0xE0 <= lead <= 0xEF else 4 if 0xF0 <= lead <= 0xF4 else 0
        )
        candidate = values[position : position + width]
        if (
            width
            and len(candidate) == width
            and all(0x80 <= byte <= 0xBF for byte in candidate[1:])
        ):
            try:
                recovered = bytes(candidate).decode("utf-8")
            except UnicodeDecodeError:
                pass
            else:
                findings.append((position, text[position : position + width], recovered))
                position += width
                continue
        position += 1
    return findings
