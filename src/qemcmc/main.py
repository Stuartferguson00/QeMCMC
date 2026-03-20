# Internal
from qemcmc.sampler import ClassicalMCMC, QeMCMC
from qemcmc.utils import ModelMaker, MCMCChain
from qemcmc.coarse_grain import CoarseGraining


# External
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm


def plot_thermalisation(chains_dict, n_spins, temp, exact_energy=None, lowest_levels=None):
    """
    chains_dict format: {"Label": (chain_data, "color")}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each algorithm's energy trajectory
    for label, (energy_data, color) in chains_dict.items():
        steps = np.arange(1, len(energy_data) + 1)
        ax.plot(steps, energy_data, label=label, color=color, linewidth=2)

    # Add reference lines if provided
    if exact_energy is not None:
        ax.axhline(exact_energy, color="black", label="Exact average energy", zorder=3)

    if lowest_levels is not None:
        for i, level in enumerate(lowest_levels):
            lbl = "Lowest energy levels" if i == 0 else ""
            ax.axhline(level, color="lightcyan", linestyle="--", label=lbl, zorder=1)

    # Match the paper's style
    ax.set_xscale("log")
    ax.set_xlabel("MCMC step (Log Scale)", fontsize=12)
    ax.set_ylabel("Average Energy", fontsize=12)
    ax.set_title(f"Repeated Coarse Graining Benchmarks ({n_spins} spins | T = {temp})", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n = 24
    steps = 2000
    reps = 10
    temp = 0.1

    model_maker = ModelMaker(n, "Coarse Grained Ising", "24 Spin Test")
    model = model_maker.model
    initial_states = model.initial_state

    cg = CoarseGraining(n)

    m_values = [3, 4, 6, 8, 12]
    colors = ["purple", "red", "pink", "green", "orange"]

    results_dict = {}

    # --- 1. Run Classical Baselines ---
    print("--- Running Classical Baselines ---")

    # Uniform
    def run_uniform(rep):
        sampler = ClassicalMCMC(model, temp, method="uniform")
        return sampler.run(steps, initial_state=initial_states[rep], verbose=True)

    uni_chains = list(tqdm(Parallel(n_jobs=-1, return_as="generator")(delayed(run_uniform)(rep) for rep in range(reps)), total=reps, desc="Running Uniform chains", leave=False))
    results_dict["Uniform"] = (np.mean([c.get_current_energy_array() for c in uni_chains], axis=0), "darkorange")

    # Local
    def run_local(rep):
        sampler = ClassicalMCMC(model, temp, method="local")
        return sampler.run(steps, initial_state=initial_states[rep], verbose=True)

    loc_chains = list(tqdm(Parallel(n_jobs=-1, return_as="generator")(delayed(run_local)(rep) for rep in range(reps)), total=reps, desc="Running Local chains", leave=False))
    results_dict["Local"] = (np.mean([c.get_current_energy_array() for c in loc_chains], axis=0), "forestgreen")

    # --- 2. Run Repeated QeMCMC ---
    print("\n--- Running Quantum Chains (Repeated CG) ---")
    m_values = [3, 4, 6, 8, 12]
    colors = ["purple", "red", "pink", "dodgerblue", "gold"]

    for m, color in tqdm(zip(m_values, colors), total=len(m_values), desc="Testing Partitions (m)"):

        def run_repeated_qemcmc(rep):
            sampler = QeMCMC(
                model=model,
                gamma=(0.3, 0.6),
                time=(2, 20),
                temp=temp,
                coarse_graining=cg,
                m=m,
            )
            return sampler.run(
                steps,
                initial_state=initial_states[rep],
                name=f"QeMCMC m={m}",
                verbose=True,
                sample_frequency=1,
            )

        chains = list(
            tqdm(
                Parallel(n_jobs=-1, return_as="generator")(delayed(run_repeated_qemcmc)(rep) for rep in range(reps)),
                total=reps,
                desc=f"Running m={m} chains",
                leave=True,
            )
        )

        all_energies = np.array([chain.get_current_energy_array() for chain in chains])
        avg_energy = np.mean(all_energies, axis=0)

        results_dict[f"QeMCMC m={m} ({n // m} spin blocks)"] = (avg_energy, color)

    exact_gs_energy = model.get_ground_state()
    plot_thermalisation(results_dict, n_spins=n, temp=temp, exact_energy=exact_gs_energy)
