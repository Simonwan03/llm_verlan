from __future__ import annotations

import re
from typing import Sequence

LETTER_CLASS = "A-Za-zÀ-ÖØ-öø-ÿŒœÆæ"


def tokenize_sentence(tokenizer, sentence: str):
    """Tokenize a sentence with offsets and special-token masks for span mapping."""
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return tokens, enc


def find_target_char_span(sentence: str, target_word: str) -> tuple[int, int]:
    """
    Find the first whole-word occurrence of `target_word` in `sentence`.

    This is more robust than matching tokenized sublists because sentence-piece
    models can tokenize standalone words and in-sentence words differently.
    """
    pattern = re.compile(
        rf"(?<![{LETTER_CLASS}]){re.escape(target_word)}(?![{LETTER_CLASS}])",
        flags=re.IGNORECASE,
    )
    match = pattern.search(sentence)
    if not match:
        raise ValueError(f"Cannot find a whole-word match for '{target_word}' in: {sentence}")
    return match.span()


def char_span_to_token_span(
    offsets: Sequence[Sequence[int]],
    special_tokens_mask: Sequence[int],
    char_span: tuple[int, int],
) -> tuple[int, int]:
    """Map a character span to a token span using tokenizer offsets."""
    start_char, end_char = char_span
    token_ids: list[int] = []

    for idx, (offset, is_special) in enumerate(zip(offsets, special_tokens_mask)):
        if is_special:
            continue
        token_start, token_end = int(offset[0]), int(offset[1])
        if token_end <= start_char:
            continue
        if token_start >= end_char:
            break
        if token_start < end_char and token_end > start_char:
            token_ids.append(idx)

    if not token_ids:
        raise ValueError(f"Failed to map character span {char_span} onto token offsets.")

    return token_ids[0], token_ids[-1] + 1


def find_target_span(tokenizer, sentence: str, target_word: str):
    """Return sentence tokens, full encoding, token span, and target subword tokens."""
    sent_tokens, enc = tokenize_sentence(tokenizer, sentence)
    char_span = find_target_char_span(sentence, target_word)
    offsets = enc["offset_mapping"][0].tolist()
    special_tokens_mask = enc["special_tokens_mask"][0].tolist()
    token_span = char_span_to_token_span(offsets, special_tokens_mask, char_span)
    start, end = token_span
    target_tokens = sent_tokens[start:end]
    return sent_tokens, enc, token_span, target_tokens, char_span
