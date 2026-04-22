import json
import os
import subprocess
import copy
import sys
import numpy as np

def run_batch():
    # Paths
    base_config_path = os.path.join(os.path.dirname(__file__), "experiment_config.json")
    temp_config_path = os.path.join(os.path.dirname(__file__), "temp_config.json")
    script_path = os.path.join(os.path.dirname(__file__), "samplingspacetimes.py")

    # Load the baseline configuration
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    base_config["temp"] = np.nan
    base_config["epsilon"] = 0.1 # Can I set to nan safely?
    base_config["sampling_frequency"]  = 1
    base_config["uniform"] = True
    base_config["initial_states"]= [
        "0",
        "1",
        "KR",
        "KR"
    ]



    # Define the parameter grids you want to loop over
    cardinalities = [36,]


    for C in cardinalities:
        print(f"========== Starting run for C={C} ==========")
        num_max_qubits = 10
        num_bits = C*(C-1)//2
        num_subgroups = np.ceil(num_bits/num_max_qubits)

        # Create a copy of the base config and modify the parameters
        config = copy.deepcopy(base_config)
        config["C"] = C
        

        #config["num_steps_q"] = 1000
        #config["num_steps_c"] = 5000
        config["num_steps_q"] = 50
        config["num_steps_c"] = 200

        # set the experiment parameters of C and temp
        config["experiments"].append({
                "name": f"qe {C}, {num_subgroups}",
                "type": "qe",
                "params": {
                    "gamma": 0.7,
                    "time": [0.5,
                                1.5],
                    "m": num_subgroups,
                    "repeated": False
                }
                })
        



        # Save to a temporary config file
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Execute the sampling script with the temporary config
        subprocess.run([sys.executable, script_path, "--config", temp_config_path], check=True)
    # Cleanup the temporary file once all experiments are done
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

if __name__ == "__main__":
    run_batch()