import argparse
import os
import sys
import pickle
import itertools
import matplotlib.pyplot as plt

# Import the plotting helper function
from CST_helpers import * #s plot_chains_BD, plot_chains_height, plot_chains_height_hist, plot_chains_num_relations_hist, plot_chains_ordering_fraction, plot_chains_minimal_elements, plot_chains_num_relations



def analyze_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        save_data = pickle.load(f)

    global_params = save_data.get('global_params', {})
    results = save_data.get('experiments', {})
    initial_states = global_params.get('initial_states', [])
    C = global_params.get('C', 'Unknown')
    temp = global_params.get('temp', 'Unknown')
    num_steps_q = global_params.get('num_steps_q', 'Unknown')

    if len(initial_states) > 0 and results:

        constr_rejs = []
        MH_rejs = []
        self_rejs = []
        total_rejs = []
        gammas = []
        times = []
        for res_str in results:
            res = results[res_str]

            constr_rej = [res['constraint rejections']["rejections_0"]]#, res['constraint rejections']["rejections_1"]]
            MH_rej = [res['MH rejections']["MH_rejections_0"]]#, res['MH rejections']["MH_rejections_1"]]
            self_rej = [res['self rejections']["self_rejections_0"]]#, res['self rejections']["self_rejections_1"]]
            total_rej = MH_rej+self_rej + constr_rej

            gamma = res['params'].get('gamma', 'Unknown')
            time = res['params'].get('time', 'Unknown')
            #scatter plot where the color is weight of constr_rej
            constr_rejs.append(np.sum(constr_rej)/(num_steps_q))
            MH_rejs.append(np.sum(MH_rej)/(num_steps_q))
            self_rejs.append(np.sum(self_rej)/(num_steps_q))
            total_rejs.append(np.sum(total_rej)/(num_steps_q))
            gammas.append(gamma)
            times.append(time)
            
        # for i in range(len(constr_rejs)):
        #     gamma = gammas[i]
        #     time = times[i]
        #     constr_rej = constr_rejs[i]

        # for each data point, plot gamma vs time and the color is the weight of constr_rej
        # fix this, so they share a color scale, and the color is the weight of constr_rej, MH_rej, self_rej and total_rej respectively
        
        cmap = plt.get_cmap('viridis')

        vmin, vmax = 0, 1
        scatter = axs[0,0].scatter(gammas, times, c=constr_rejs, vmin=vmin, vmax=vmax, cmap=cmap, s=100, alpha =1)
        #axs[0,0].colorbar(scatter, label='Constraint Rejection Rate')
        axs[0,0].set_title('Constraint Rejection Rate')

        scatter = axs[0,1].scatter(gammas, times, c=MH_rejs, vmin=vmin, vmax=vmax, cmap=cmap  , s=100, alpha =1)
        #axs[0,1].colorbar(scatter, label='MH Rejection Rate')
        axs[0,1].set_title('MH Rejection Rate')

        scatter = axs[1,0].scatter(gammas, times, c=self_rejs, vmin=vmin, vmax=vmax, cmap=cmap, s=100, alpha =1)
        #axs[1,0].colorbar(scatter, label='Self Rejection Rate')
        axs[1,0].set_title('Self Rejection Rate')

        scatter = axs[1,1].scatter(gammas, times, c=total_rejs, vmin=vmin, vmax=vmax, cmap=cmap, s=100, alpha =1)
        #axs[1,1].colorbar(scatter, label='Total Rejection Rate')
        axs[1,1].set_title('Total Rejection Rate')

        # Only do this once
        print("lowest total rejectsion rate: ", np.min(total_rejs))
        print("found at gamma: ", gammas[np.argmin(total_rejs)], " and time: ", times[np.argmin(total_rejs)])
        



        """# for all unique values of time, plot the constr_rej, MH_rej and self_rej against gamma
        unique_times = set(times)
        unique_times = sorted(unique_times)
        fig, axs = plt.subplots(len(unique_times), 1, figsize=(10, 5*len(unique_times)))
        if type(axs) != np.ndarray:
            axs = [axs]
        for i, t in enumerate(unique_times):
            gamma = [gammas[i] for i in range(len(times)) if times[i] == t]
            constr_rej = [constr_rejs[i] for i in range(len(times)) if times[i] == t]
            MH_rej = [MH_rejs[i] for i in range(len(times)) if times[i] == t]
            self_rej = [self_rejs[i] for i in range(len(times)) if times[i] == t]
            total_rej = [total_rejs[i] for i in range(len(times)) if times[i] == t]

            axs[i].plot(gamma, total_rej, label='Total Rejections')
            axs[i].plot(gamma, constr_rej, label='Constraint Rejections')
            axs[i].plot(gamma, MH_rej, label='MH Rejections')
            axs[i].plot(gamma, self_rej, label='Self Rejections')
            axs[i].set_title(f'Rejection Rates for time={t}')
            axs[i].set_yscale('log')
            axs[i].set_ylim(1e-1 , 1)

        fig.suptitle(f'Rejection Rates for C={C}, temp={temp}')
        fig.supxlabel('Gamma')
        fig.supylabel('Rejection rate')
        # get rid of x ticks on each axs except the last one
        for i in range(len(unique_times)-1):
            axs[i].set_xticks([])

        plt.tight_layout()
        # add space between axis
        
        plt.subplots_adjust(hspace=0.3)
        plt.legend()
        plt.show()"""

        


        return scatter

       
        
    else:
        print("No valid chain data found to plot.")

if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), "hyperparameters_saved_chains")
    
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
    

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    #analyze_results(file)
    for _dir in exp_dirs:
        file_path = os.path.join(folder_path, _dir, "data.pkl")


        file = str(file_path)
        
        scatter = analyze_results(file)
    cmap = plt.get_cmap('viridis') 
    scatter = axs[1,1].scatter([0.1,0.1], [1,1], c=[0,1], cmap=cmap, s=0, alpha = 1)
    # force colorbar to be between 0 and 1
    fig.colorbar(scatter, ax=axs, label='Rejection Rate')
    fig.suptitle(f'Rejection Rates')
    fig.supxlabel('Gamma')
    fig.supylabel('Time')
    plt.title(f'Rejection Rates')
    plt.show()

        # for _dir in exp_dirs:
    #     file_path = os.path.join(folder_path, _dir, "data.pkl")


    #     file = str(file_path)
        
    #     analyze_results(file)
