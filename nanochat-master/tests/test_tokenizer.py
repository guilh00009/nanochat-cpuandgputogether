
import pytest
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer, HuggingFaceTokenizer

def test_special_tokens_list():
    assert "<|bos|>" in SPECIAL_TOKENS
    assert "<|user_start|>" in SPECIAL_TOKENS
    assert "<|assistant_end|>" in SPECIAL_TOKENS

def test_rustbpe_tokenizer_structure():
    # We can't easily test functionality without the compiled rust extension,
    # but we can check class structure if we mock or if it's importable.
    # Assuming runtime environment has it or we just check API existence.
    assert hasattr(RustBPETokenizer, 'train_from_iterator')
    assert hasattr(RustBPETokenizer, 'encode')
    assert hasattr(RustBPETokenizer, 'decode')

def test_huggingface_tokenizer_structure():
    assert hasattr(HuggingFaceTokenizer, 'train_from_iterator')
    assert hasattr(HuggingFaceTokenizer, 'encode')
    assert hasattr(HuggingFaceTokenizer, 'decode')
