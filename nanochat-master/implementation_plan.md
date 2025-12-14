# Fix Intel XPU Dependency Resolution

This plan addresses the `uv sync` failure for `xpu` extra caused by a missing transitive dependency `pytorch-triton-xpu`.

## User Review Required

> [!IMPORTANT]
> The fix assumes `pytorch-triton-xpu` is available in the `https://download.pytorch.org/whl/xpu` index. This is standard for PyTorch XPU builds.

## Proposed Changes

### Configuration
#### [MODIFY] [pyproject.toml](file:///c:/Users/kelle/Downloads/nanochat-master/nanochat-cpuandgputogether/nanochat-master/pyproject.toml)
- Add a source mapping for `pytorch-triton-xpu` to the `pytorch-xpu` index.
- Since the index is marked `explicit = true`, transitive dependencies like `pytorch-triton-xpu` must be manually mapped or the index must be non-explicit. Mapping is safer to avoid polluting other resolutions.

```toml
[tool.uv.sources]
# ... existing sources ...
pytorch-triton-xpu = [
    { index = "pytorch-xpu", extra = "xpu" },
]
```
*Note: We can map it generally or scoped. Since it's unique to XPU, mapping it to the index is sufficient.*

## Verification Plan

### Manual Verification
- User runs `uv sync --extra xpu` again.
- If successful, the error "pytorch-triton-xpu was not found" should disappear.
