# Simulating Coupled Harmonic Oscillators

### Stucture of code.

<p></p>

**class Molecule()**: This class is essential for formulating the problem. It will hold information about the given molecule to simulate. For example, the class can be initialized by providing the molecule (probably as a .pdb file from (https://www.rcsb.org/)). It can be initialized by whether one wants to use only C-alpha atoms or the whole atomic description, whether the initial positions should be random etc.<br>

The necessary functions it must include are:<br>
  get_spring_matrix()<br>
  get_initial_positions()<br>
  get_initial_velocities()<br>
  get_masses()<br>

_Responsible for implementing: Ioannis and Adi._

<p></p> 

**class Oracles()**: This class is essential for getting the corresponding quantum circuits for oracles used in the algorithm.<br>

The necessary function it must include are:<br>
  get_K_oracle()<br>
  get_S_oracle()<br> 
  get_M_oracle()<br> 
  get_V_oracle()<br> 

_Responsible for implementing: Ioannis and Adi_

<p></p>

**class PrepareInitialState()**: This class uses the oracle and molecule class to prepare the initial state. <br>

The necessary functions in this class are:<br>
  def inequality_testing()<br>

_Responsible for implementing: Ioannis and Adi_

<p></p>

## Development Setup and Guidelines

### Initial Setup

1.  **Install uv (if you haven't already):**
    `uv` is an extremely fast Python package installer and resolver, written in Rust, intended as a drop-in replacement for `pip` and `pip-tools`.
    Follow the official installation instructions for `uv` from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv). For example, on macOS and Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create/Sync Virtual Environment:**
    Navigate to the project root directory and run:
    ```bash
    uv sync
    ```
    This will create a `.venv` folder with all the dependencies specified in `pyproject.toml`.

3.  **Install ProDy:**
    ProDy can sometimes have build issues with newer packaging tools or specific environments. If `uv sync` doesn't install it correctly or if you encounter issues, try installing it with `pip` within the `uv` managed environment:
    ```bash
    # Activate the virtual environment if not already active
    # source .venv/bin/activate (for Linux/macOS)
    # .venv\Scripts\activate (for Windows)
    pip install prody==2.4.1
    ```
    *Note: `prody==2.4.1` is not compatible with NumPy 2.0+. If you encounter issues, you might need to adjust your `numpy` version in `pyproject.toml` to `numpy~=1.26.4` and compatible versions for `scipy` and `matplotlib`.*

### Development Workflow

-   **VS Code Extension:** Please install the "ruff" extension for VS Code. `Ruff` is an extremely fast Python linter and formatter, written in Rust.
-   **Linting and Formatting:** Before publishing your branch, run:
    ```bash
    uv run ruff check --fix
    uv run ruff format
    ```
-   **Branching:** Create a new branch for every feature you develop.
-   **Testing:**
    -   Add tests under the `tests/` folder.
    -   Run your tests with:
        ```bash
        uv run pytest tests
        ```
-   **Merging to `main`:**
    1.  Submit a pull request.
    2.  Assign a review.
    3.  Reference an issue.


