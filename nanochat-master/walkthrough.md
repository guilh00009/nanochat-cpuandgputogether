# Verification: Intel GPU Support Addition

I have added support for Intel GPUs (XPU) to the project's installation configuration and documentation.

## Changes Verified

### 1. `pyproject.toml`
- **Source Added**: `pytorch-xpu` index pointing to `https://download.pytorch.org/whl/xpu`.
- **Dependency Added**: `xpu` extra requiring `torch>=2.8.0`.
- **Conflicts Updated**: Added `xpu` to the mutual exclusion list with `cpu` and `cuda`.

```toml
[tool.uv.sources]
torch = [  
    { index = "pytorch-cpu", extra = "cpu" },  
    { index = "pytorch-cu128", extra = "cuda" },  
    { index = "pytorch-xpu", extra = "xpu" },
]

[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true
```

### 2. `INSTALL.md`
- Added **"For Intel GPUs (XPU Support)"** section with the command:
  ```bash
  uv sync --extra xpu
  ```

## Manual Verification Steps
To fully verify the installation on a machine with an Intel Arc GPU:
1.  Ensure prerequisites (drivers, Visual C++, OneAPI if needed) are met.
2.  Run: `uv sync --extra xpu`
3.  Verify PyTorch detects XPU:
    ```bash
    uv run python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}')"
    ```
