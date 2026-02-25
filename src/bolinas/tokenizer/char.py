"""DNA character-level tokenizer with BOS and EOS tokens.

Creates a HuggingFace-compatible tokenizer that can be saved and loaded
via ``AutoTokenizer.from_pretrained`` with no custom code required.
"""

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

DNA_BASES = "acgt"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def _build_char_vocab() -> dict[str, int]:
    """Build vocabulary mapping DNA bases to integer IDs.

    IDs: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, then a=4, c=5, g=6, t=7.
    All lowercase to match the Lowercase() normalizer.
    """
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    offset = len(SPECIAL_TOKENS)
    for i, base in enumerate(DNA_BASES):
        vocab[base] = offset + i
    return vocab


def _build_backend_tokenizer() -> Tokenizer:
    """Assemble a Rust-backed tokenizer with serializable components."""
    vocab = _build_char_vocab()
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.normalizer = Lowercase()
    backend.pre_tokenizer = Split(pattern=Regex(".{1}"), behavior="isolated")
    backend.decoder = Fuse()
    backend.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])],
    )
    return backend


def create_char_tokenizer() -> PreTrainedTokenizerFast:
    """Create a DNA character-level tokenizer backed by HuggingFace's Rust engine.

    The returned ``PreTrainedTokenizerFast`` can be saved with
    ``tok.save_pretrained(path)`` and reloaded anywhere via
    ``AutoTokenizer.from_pretrained(path)`` — no ``bolinas`` dependency
    or ``trust_remote_code`` needed.
    """
    backend = _build_backend_tokenizer()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
