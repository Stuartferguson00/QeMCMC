import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from qemcmc.coarse_grain import CoarseGraining
from qemcmc.model import ModelMaker
from qemcmc.sampler import ClassicalProposal, QeProposal
from qemcmc.sampler.runners import MCMCRunner
from qemcmc.utils import plot_chains

N_SPINS = 16
STEPS = 150
REPS = 10
TEMP = 0.1
M_BLOCKS = 4  # coarse-grained QeMCMC: N_SPINS // M_BLOCKS spins simulated per partition

# Standard QeMCMC simulates all N_SPINS qubits at once, so its cost is exponential in N_SPINS
# Set to False to compare only the arms that scale.
RUN_STANDARD_QEMCMC = True

console = Console()


def run_chain_with_seed(seed, runner, **kwargs):
    np.random.seed(seed)
    return runner.run(**kwargs)


class ProgressProposal:
    """
    Wraps a proposal so each chain records its hop count to its own file.

    ``MCMCRunner.run`` only ever calls ``proposer.update()``, so this needs no cooperation
    from the runner. Chains run in separate processes, so a file is the simplest thing they
    can all write to without the parent having to coordinate them.
    """

    def __init__(self, proposer, progress_file):
        self._proposer = proposer
        self._progress_file = Path(progress_file)
        self._hops = 0

    def update(self, current_state: str) -> str:
        next_state = self._proposer.update(current_state)
        self._hops += 1
        self._progress_file.write_text(str(self._hops))
        return next_state


class ChainProgress(Progress):
    """One bar per chain, re-read from the workers' files on every refresh."""

    def __init__(self, progress_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_dir = Path(progress_dir)

    def get_renderable(self):
        for task in self.tasks:
            progress_file = self.progress_dir / str(task.id)
            if progress_file.exists():
                try:
                    task.completed = int(progress_file.read_text() or 0)
                except ValueError:
                    pass  # caught mid-write; the next refresh picks it up
        return super().get_renderable()


def run_arm(label, color, proposer, runner, initial_states, seeds):
    """Run REPS independent chains for one proposal method and add them to the current plot."""
    console.print(f"[bold]{label}[/bold]")
    start = time.time()

    with tempfile.TemporaryDirectory() as progress_dir:
        columns = (
            TextColumn("  {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("hops"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        with ChainProgress(progress_dir, *columns) as progress:
            task_ids = [progress.add_task(f"chain {i}", total=STEPS) for i in range(len(seeds))]
            chains = Parallel(n_jobs=-1)(
                delayed(run_chain_with_seed)(
                    seed,
                    runner,
                    proposer=ProgressProposal(proposer, Path(progress_dir) / str(task_ids[i])),
                    n_hops=STEPS,
                    initial_state=initial_states[i],
                    verbose=False,
                )
                for i, seed in enumerate(seeds)
            )

    console.print(f"  [dim]{label}: {time.time() - start:.1f} s[/dim]")
    plot_chains(chains, color, label=label, plot_individual_chains=True)
    return chains


def annotate_below_legend(ax, legend, text):
    """Place text directly under the legend box, in axes coordinates."""
    ax.figure.canvas.draw()  # the legend has no extent until the figure is drawn
    bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bbox.x0, bbox.y0 - 0.03, text, transform=ax.transAxes, ha="left", va="top", fontsize=9)


if __name__ == "__main__":
    np.random.seed(2)
    start_time = time.time()

    model = ModelMaker(N_SPINS, model_type="Fully Connected Ising", name=f"{N_SPINS} Spin Ising").model
    runner = MCMCRunner(model=model, temp=TEMP)

    initial_states = ["".join(np.random.choice(["0", "1"], size=N_SPINS)) for _ in range(REPS)]
    seeds = np.arange(REPS)

    cg = CoarseGraining(n=N_SPINS)

    run_arm("Classical uniform MCMC", "orange", ClassicalProposal(model, method="uniform"), runner, initial_states, seeds)
    run_arm("Classical local MCMC", "green", ClassicalProposal(model, method="local"), runner, initial_states, seeds)
    run_arm(
        f"Coarse-Grained QeMCMC (m={M_BLOCKS}, {N_SPINS // M_BLOCKS} spin blocks)",
        "blue",
        QeProposal(model=model, gamma=(0.3, 0.6), time=(1, 20), coarse_graining=cg, m=M_BLOCKS),
        runner,
        initial_states,
        seeds,
    )

    if RUN_STANDARD_QEMCMC:
        run_arm("Standard QeMCMC", "red", QeProposal(model=model, gamma=(0.3, 0.6), time=(1, 20)), runner, initial_states, seeds)

    ax = plt.gca()
    ax.set_title(f"QeMCMC thermalisation ({N_SPINS} spins | T = {TEMP})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    legend = ax.legend(loc="upper right", fontsize=9)
    annotate_below_legend(ax, legend, f"Time taken: {time.time() - start_time:.2f} seconds")
    plt.tight_layout()
    plt.show()
