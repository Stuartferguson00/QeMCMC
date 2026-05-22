# Installation

Install the latest release from [PyPI](https://pypi.org/project/qemcmc/) (requires Python 3.13+):

```bash
pip install qemcmc
```

## From source

The example notebooks live in the [repo](https://github.com/Stuartferguson00/QeMCMC), not the published package. To run them — or to develop QeMCMC — clone the repo and install with [`uv`](https://astral.sh/uv):

```bash
git clone https://github.com/Stuartferguson00/QeMCMC.git
cd QeMCMC
uv sync
```

`uv sync` creates a local `.venv` and installs the locked dependencies from `pyproject.toml` and `uv.lock`. If you don't have `uv`, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
