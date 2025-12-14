# Add Intel GPU Support to Installation

This plan outlines the changes required to support Intel GPUs (XPU) in the project's installation process using `uv`.

## User Review Required

> [!IMPORTANT]
> This plan assumes that the standard PyTorch XPU wheels are available at `https://download.pytorch.org/whl/xpu` for the version `2.8.0` (or compatible). Please confirm if a specific version or index URL is preferred.

## Proposed Changes

### Configuration
#### [MODIFY] [pyproject.toml](file:///c:/Users/kelle/Downloads/nanochat-master/nanochat-cpuandgputogether/nanochat-master/pyproject.toml)
- Add `xpu` to `[project.optional-dependencies]`.
- Add `pytorch-xpu` index definition pointing to `https://download.pytorch.org/whl/xpu`.
- Add source mapping for `xpu` extra to `pytorch-xpu` index.
- Update `tool.uv.conflicts` to strictly separate `cpu`, `cuda`, and `xpu`.

### Documentation
#### [MODIFY] [INSTALL.md](file:///c:/Users/kelle/Downloads/nanochat-master/nanochat-cpuandgputogether/nanochat-master/INSTALL.md)
- Add a new section **"For Intel GPUs (XPU Support)"**.
- Provide the installation command: `uv sync --extra xpu`.

## Verification Plan

### Automated Verification
- Verify `pyproject.toml` is valid TOML.
- Attempt to lock dependencies (dry-run if possible, or inspection) - *Note: Actual installation might fail if the user doesn't have the hardware/drivers or if the future version 2.8.0 isn't available yet, but we will verify the configuration syntax.*

### Manual Verification
- The user can run `uv sync --extra xpu` to confirm the resolver finds the packages.
