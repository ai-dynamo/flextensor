<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Installation

## Requirements

- Python 3.11 or higher
- PyTorch 2.5 or higher (with CUDA support)
- CUDA-capable GPU

Python 3.14 requires PyTorch 2.10 or higher.

## Install from Package

FlexTensor is available on PyPI. Use the following command to install:

```bash
pip install flextensor
```

## Install from Source

Clone the repository and install with:

```bash
git clone https://github.com/ai-dynamo/flextensor.git
cd flex-tensor
pip install .
```

## Development Installation

For development, install with all dependency groups:

```bash
uv venv
uv pip install --group all -e .
```

This requires uv 0.6.7 or newer. If you use pip directly instead, use pip
25.1 or newer for `pip install --group`.

This includes:

- Testing tools (pytest, pytest-cov)
- Pre-commit hooks
- Documentation generation tools (mkdocs, mike)

## Verify Installation

```python
import flextensor
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"FlexTensor version: {flextensor.__version__}")
```
