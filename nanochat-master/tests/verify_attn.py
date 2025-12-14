
import torch
import torch.nn.functional as F
import math

def manual_attention(q, k, v, enable_gqa, n_head, n_kv_head, kv_cache=None, Tq=None, Tk=None):
    # Replicating the logic added to gpt.py
    if enable_gqa:
        # Manual GQA: repeat the key/value heads
        k = k.repeat_interleave(n_head // n_kv_head, dim=1)
        v = v.repeat_interleave(n_head // n_kv_head, dim=1)
    
    # Scaled dot product attention
    # (B, H, Tq, D) @ (B, H, D, Tk) -> (B, H, Tq, Tk)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
    if kv_cache is None or Tq == Tk:
        # Causal mask: mask out future tokens (upper triangle excluding diagonal)
        # In this test Tq == Tk usually for simple causal case
        mask = torch.triu(torch.ones(Tq, Tk, dtype=torch.bool, device=q.device), diagonal=1)
        att = att.masked_fill(mask, float('-inf'))
    elif Tq == 1:
        # Single query attends to all keys - no masking needed
        pass
    else:
        # Chunked inference with prefix
        # Construct mask where True = keep, then invert for masked_fill
        mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
        prefix_len = Tk - Tq
        if prefix_len > 0:
            mask[:, :prefix_len] = True
        mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
        att = att.masked_fill(~mask, float('-inf'))

    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def test_equivalence():
    torch.manual_seed(42)
    B = 2
    H = 4 # n_head
    H_kv = 4 # n_kv_head (no GQA for first test)
    D = 16
    T = 10
    
    # Case 1: Standard Causal Attention (Training style)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H_kv, T, D)
    v = torch.randn(B, H_kv, T, D)
    
    # Reference
    ref_y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Manual
    man_y = manual_attention(q, k, v, enable_gqa=False, n_head=H, n_kv_head=H_kv, kv_cache=None, Tq=T, Tk=T)
    
    print(f"Case 1 (Standard Causal) Max diff: {(ref_y - man_y).abs().max().item()}")
    assert torch.allclose(ref_y, man_y, atol=1e-5), "Case 1 Failed"

    # Case 2: GQA
    H = 4
    H_kv = 2
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H_kv, T, D)
    v = torch.randn(B, H_kv, T, D) # 2 heads
    
    # Reference
    ref_y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    
    # Manual
    man_y = manual_attention(q, k, v, enable_gqa=True, n_head=H, n_kv_head=H_kv, kv_cache=None, Tq=T, Tk=T)
    
    print(f"Case 2 (GQA) Max diff: {(ref_y - man_y).abs().max().item()}")
    # Tolerance might need to be slightly higher for float arithmetic accumulation differences
    assert torch.allclose(ref_y, man_y, atol=1e-5), "Case 2 Failed"

    print("All tests passed!")

if __name__ == "__main__":
    test_equivalence()
