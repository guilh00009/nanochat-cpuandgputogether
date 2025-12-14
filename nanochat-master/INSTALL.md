# How to Install Requirements for Nanochat

This project uses **uv** for dependency management and **Rust** for its high-performance tokenizer (`rustbpe`). Follow these steps to set up your environment.

## 1. Prerequisites

### Install `uv`
`uv` is an extremely fast Python package installer and resolver.
**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Rust
The project requires compiling the `rustbpe` extension.
1.  Visit [rustup.rs](https://rustup.rs/) and download the installer.
2.  Run the installer and follow the on-screen instructions.
3.  Ensure `cargo` is in your system PATH (restart your terminal after installation).

### Python 3.10+
Ensure you have Python 3.10 or later installed.
```bash
python --version
```

## 2. Installation

Navigate to the project root directory in your terminal.

```bash
cd path/to/nanochat
```

### Option A: Using `uv` (Recommended)

`uv` will automatically create a virtual environment (`.venv`), resolve dependencies from `uv.lock`, and install the project.

**For NVIDIA GPUs (CUDA Support):**
```bash
uv sync --extra cuda
```

**For Intel GPUs (XPU Support):**
```bash
uv sync --extra xpu
```

**For CPU Only:**
```bash
uv sync --extra cpu
```

### Option B: Using Standard `pip` (Legacy)

If you strictly prefer standard `pip` and `venv`:

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

2.  **Activate it:**
    *   Windows: `.venv\Scripts\activate`
    *   Linux/Mac: `source .venv/bin/activate`

3.  **Install dependencies:**
    *   Note: You may need to manually install the correct PyTorch version for your hardware first (see [pytorch.org](https://pytorch.org/get-started/locally/)).
    ```bash
    pip install -e .
    ```

## 3. Verification

Activate the environment (if not already active) and run a quick test.

**Using `uv`:**
`uv` allows you to run commands inside the environment without manually activating it:
```bash
uv run python -c "import torch; print(f'Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

**Using activated shell:**
```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -c "import torch; print(f'Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

## 4. Compile `rustbpe` (if needed)

The `rustbpe` tokenizer should be automatically compiled and installed by `uv sync` via `maturin`. If you need to rebuild it manually:

```bash
uv run maturin develop --release -m rustbpe/Cargo.toml
```
