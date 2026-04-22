import argparse
import os
import sys
import pickle
import itertools
import matplotlib.pyplot as plt

# Import the plotting helper function
from CSThelpers import * #s plot_chains_BD, plot_chains_height, plot_chains_height_hist, plot_chains_num_relations_hist, plot_chains_ordering_fraction, plot_chains_minimal_elements, plot_chains_num_relations



import numpy as np

def calculate_mcmc_autocorrelation(chain: MCMCChain, observable_fn, max_lag: int = None):
    """
    Calculates the autocorrelation function of an observable over an MCMC chain.
    
    Args:
        chain: An instance of MCMCChain.
        observable_fn: A function that takes a bitstring and returns a scalar value.
        max_lag: The maximum lag to compute. Defaults to len(chain)//2.
        
    Returns:
        lags: Array of lag indices.
        autocorr: Array of autocorrelation values corresponding to the lags.
    """
    # 1. Extract the actual trajectory (the Markov Chain)
    # We use get_list_markov_chain() to ensure we account for rejected 
    # steps where the state remains the same.
    trajectory_bitstrings = chain.get_list_markov_chain()
    
    # 2. Map the observable function over the trajectory
    data = np.array([observable_fn(bitstring_to_matrix(s)) for s in trajectory_bitstrings])
    
    n = len(data)
    if max_lag is None:
        max_lag = n // 10
        
    # 3. Center the data
    data_centered = data - np.mean(data)
    
    # 4. Compute autocorrelation using FFT (Wiener-Khinchin Theorem)
    # Pad to power of 2 for performance and to avoid cyclic correlation artifacts
    f_size = 2 ** (n * 2 - 1).bit_length()
    fft_res = np.fft.fft(data_centered, n=f_size)
    spectral_density = fft_res * np.conj(fft_res)
    autocorr_full = np.real(np.fft.ifft(spectral_density))
    
    # 5. Normalize by variance (autocorr at lag 0)
    if autocorr_full[0] < 1e-15:
        return np.arange(max_lag), np.zeros(max_lag)
        
    autocorr = autocorr_full[:max_lag] / autocorr_full[0]
    lags = np.arange(max_lag)
    
    return lags, autocorr





def plot_metric_autocorrelation(C, temp, results, initial_states, file_path, metric_func, metric_name, filename_prefix, colors, exact = None):
    plt.figure(figsize=(10, 6))
    plt.title(f"{metric_name} Autocorrelation (C={C}, T={temp})")


    if exact is not None:

        plt.scatter(exact[0], exact[1], color='k', label=f"Exact {metric_name}")

    colors = ["lightgreen", "lightblue"]
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
    for j, (name, exp_data) in enumerate(results.items()):
        print(name)
        
        chains = [exp_data['results'].get(f'chain_{i}') for i in range(len(initial_states))]
        # if all(c is not None for c in chains):
        #     plot_func(chains, label=f"{name} Proposal", color=next(color_cycle), plot_individual_chains=False)
        for i, chain in enumerate(chains):
            if chain is not None:
                
                lags, autocorr = calculate_mcmc_autocorrelation(chain, metric_func, max_lag=10)
                plt.plot(lags, autocorr, label=f"{name} Chain {i}", color=colors[j], marker=markers[i])
    
    plt.ylabel(metric_name + " autocorrelation")
    plt.xlabel("MCMC Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_save_path = os.path.join(os.path.dirname(file_path), f"{filename_prefix}_autocorrelation.png")
    plt.savefig(plot_save_path, dpi=300)    
    print(f"Plot saved successfully to {plot_save_path}")
    plt.close()





def plot_metric_histogram(C, temp, results, initial_states, file_path, plot_func, metric_name, filename_prefix, colors, exact = None):
    plt.figure(figsize=(10, 6))
    plt.title(f"{metric_name} Histogram (C={C}, T={temp})")


    if exact is not None:

        plt.scatter(exact[0], exact[1], color='k', label=f"Exact {metric_name}")

    color_cycle = itertools.cycle(colors)   
    for name, exp_data in results.items():
        print(name)
        if True:
            chains = [exp_data['results'].get(f'chain_{i}') for i in range(len(initial_states))]
            if all(c is not None for c in chains):
                plot_func(chains, label=f"{name} Proposal", color=next(color_cycle), plot_individual_chains=False)
        else:
            pass
        
    plt.ylabel(metric_name)
    plt.xlabel("MCMC Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_save_path = os.path.join(os.path.dirname(file_path), f"{filename_prefix}_hist.png")
    plt.savefig(plot_save_path, dpi=300)    
    print(f"Plot saved successfully to {plot_save_path}")
    plt.close()


def plot_metric_graph(C, temp, results, initial_states, file_path, plot_func, metric_name, filename_prefix, colors):
    plt.figure(figsize=(10, 6))
    plt.title(f"{metric_name} vs steps (C={C}, T={temp})")
    color_cycle = itertools.cycle(colors)
    
    for name, exp_data in results.items():
        chains = [exp_data['results'].get(f'chain_{i}') for i in range(len(initial_states))]
        if all(c is not None for c in chains):
            plot_func(chains, label=f"{name} Proposal", color=next(color_cycle), plot_individual_chains=False, samp_freq = 1)
            
    plt.ylabel(metric_name)
    plt.xlabel("MCMC Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_save_path = os.path.join(os.path.dirname(file_path), f"{filename_prefix}_graph.png")
    plt.savefig(plot_save_path, dpi=300)
    print(f"Plot saved successfully to {plot_save_path}")
    plt.close()

def plot_cumulative_metric_graph(C, temp, results, initial_states, file_path, plot_func, metric_name, filename_prefix, colors):
    plt.figure(figsize=(10, 6))
    plt.title(f"{metric_name} vs steps (C={C}, T={temp})")
    color_cycle = itertools.cycle(colors)
    
    for name, exp_data in results.items():
        chains = [exp_data['results'].get(f'chain_{i}') for i in range(len(initial_states))]
        if all(c is not None for c in chains):
            plot_func(chains, label=f"{name} Proposal", color=next(color_cycle), plot_individual_chains=True, samp_freq = 1)
            
    plt.ylabel(metric_name)
    plt.xlabel("MCMC Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xscale('log')
    
    plot_save_path = os.path.join(os.path.dirname(file_path), f"{filename_prefix}_cumulative_graph.png")
    plt.savefig(plot_save_path, dpi=300)
    print(f"Plot saved successfully to {plot_save_path}")
    plt.close()


def analyze_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        save_data = pickle.load(f)

    global_params = save_data.get('global_params', {})
    results = save_data.get('experiments', {})
    print("Global Params:", global_params)
    
    initial_states = global_params.get('initial_states', [])

    C = global_params.get('C', 'Unknown')
    temp = global_params.get('temp', 'Unknown')

    if len(initial_states) > 0 and results:
        colors = ["green", "orange", "blue", "pink", "purple", "red", "brown", "cyan"]


        # Load exact results for comparison
        exact_h = None
        exact_nr = None
        exact_results_path = os.path.join(os.path.dirname(__file__), f"save_files/exact_results_{C}.pkl")
        if os.path.exists(exact_results_path):
            with open(exact_results_path, 'rb') as f:
                exact_results = pickle.load(f)
                exact_h = exact_results.get("height_histogram", None)
                print(exact_h)

                exact_nr = exact_results.get("num_relations_histogram", None)
                print(exact_nr)



        #plot_metric_histogram(C, temp, results, initial_states, file_path, plot_chains_num_relations_hist, "Number of Relations", "Num_Relations", colors, exact = exact_nr)
        #plot_metric_histogram(C, temp, results, initial_states, file_path, plot_chains_height_hist, "Height", "Height", colors, exact = exact_h)
        #BD Action", "BD_action", colors)
        # plot_metric_graph(C, temp, results, initial_states, file_path, plot_chains_BD, "BD Actios", "BD_action", colors)
        
        plot_metric_autocorrelation(C, temp, results, initial_states, file_path, height, "Heights", "Height", colors)
        plot_metric_graph(C, temp, results, initial_states, file_path, plot_chains_height, "Heights", "Height", colors)
        plot_cumulative_metric_graph(C, temp, results, initial_states, file_path, plot_chains_mean_height, "Heights", "Height", colors)

        # plot_metric_graph(C, temp, results, initial_states, file_path, plot_chains_ordering_fraction, "Ordering Fractios", "Ordering_Fraction", colors)
        # plot_metric_graph(C, temp, results, initial_states, file_path, plot_chains_minimal_elements, "Minimal Elements", "Minimal_Elements", colors)
        # plot_metric_graph(C, temp, results, initial_states, file_path, plot_chains_num_relations, "Number of Relations", "Num_Relations", colors)
    else:
        print("No valid chain data found to plot.")

if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), "saved_chains")
    
    # search folder for most recent experiment directory
    if os.path.exists(folder_path):
        exp_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    else:
        exp_dirs = []
        
    if not exp_dirs:
        print("No experiment folders found in saved_chains.")
        sys.exit(1)
        
    latest_dir = max(exp_dirs, key=lambda d: os.path.getctime(os.path.join(folder_path, d)))
    print(f"Most recent experiment found: {latest_dir}")


    file = str(os.path.join(folder_path,latest_dir, "data.pkl"))
        
    analyze_results(file)

    # for _dir in exp_dirs:
    #     file_path = os.path.join(folder_path, _dir, "data.pkl")


    #     file = str(file_path)
        
    #     analyze_results(file)