"""
BPE Tokenizer using HuggingFace's `tokenizers` library.

This uses standard byte-level BPE (the same approach as GPT-2/GPT-3/GPT-4).
Each byte (0-255) is mapped to a Unicode character, and merges happen between
these byte-level tokens. The Rust backend makes training and encoding very fast.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC


def create_and_train(
    corpus_lines: list[str],
    vocab_size: int = 10000,
    min_frequency: int = 2,
) -> Tokenizer:
    """
    Create and train a byte-level BPE tokenizer.

    Args:
        corpus_lines: List of text lines to train on.
        vocab_size: Target vocabulary size (including base 256 byte tokens).
        min_frequency: Minimum pair frequency to consider for a merge.

    Returns:
        Trained HuggingFace Tokenizer.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        min_frequency=min_frequency,
        show_progress=False,
    )

    tokenizer.train_from_iterator(corpus_lines, trainer=trainer)
    return tokenizer


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode a full text string into token IDs."""
    output = tokenizer.encode(text)
    return output.ids


def get_stats(tokenizer: Tokenizer, text: str) -> tuple[int, int]:
    """
    Get token count and unique token count for a text.
    Returns (total_tokens, unique_tokens).
    """
    ids = tokenizer.encode(text).ids
    return len(ids), len(set(ids))
