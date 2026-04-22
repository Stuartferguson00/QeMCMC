import argparse
import os
import sys
import pickle
import itertools
import matplotlib.pyplot as plt

# Import the plotting helper function
from CSThelpers import * #s plot_chains_BD, plot_chains_height, plot_chains_height_hist, plot_chains_num_relations_hist, plot_chains_ordering_fraction, plot_chains_minimal_elements, plot_chains_num_relations

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