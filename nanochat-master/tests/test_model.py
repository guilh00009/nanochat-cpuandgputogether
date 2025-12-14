
import torch
import pytest
from nanochat.gpt import GPT, GPTConfig

def test_gpt_config_defaults():
    config = GPTConfig()
    assert config.n_layer == 12
    assert config.n_head == 6
    assert config.n_embd == 768

def test_gpt_init():
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        vocab_size=1000,
        sequence_len=128
    )
    model = GPT(config)
    
    # Check if parameters are initialized
    assert hasattr(model, 'transformer')
    assert hasattr(model, 'lm_head')
    
    # Check approximate parameter count
    # n_embd=64, vocab=1000 -> wte ~ 64000
    # lm_head ~ 64000
    # 2 layers ...
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 100000

def test_gpt_forward_pass():
    config = GPTConfig(
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=32,
        vocab_size=100,
        sequence_len=16
    )
    model = GPT(config)
    model.init_weights()
    
    # Create dummy input (batch_size=2, seq_len=8)
    idx = torch.randint(0, 100, (2, 8))
    
    # Forward pass
    logits = model(idx)
    
    assert logits.shape == (2, 8, 100)

def test_gpt_estimate_flops():
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        vocab_size=1000,
        sequence_len=128
    )
    model = GPT(config)
    flops = model.estimate_flops()
    assert flops > 0
