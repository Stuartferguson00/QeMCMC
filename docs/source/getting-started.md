# Installation

This project uses [`uv`](https://astral.sh/uv), an extremely fast Python package installer written in Rust, intended as a drop-in replacement for `pip` and `pip-tools`. Official installation instructions available at [astral.sh/uv](https://astral.sh/uv)

1.  **Install `uv`:**
    For macOS and Linux run:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a virtual environment:**
    From the project's root directory, run:
    ```bash
    uv sync
    ```
    This will create a local `.venv` folder and install all required dependencies from `pyproject.toml` and `uv.lock`.
