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
    script_path = os.path.join(os.path.dirname(__file__), "sampling_spacetimes.py")

    # Load the baseline configuration
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # Define the parameter grids you want to loop over
    temperatures = [1000,]
    cardinalities = [20,]

    for C in cardinalities:
        for temp in temperatures:
            print(f"========== Starting run for C={C}, temp={temp} ==========")
            
            # Create a copy of the base config and modify the parameters
            config = copy.deepcopy(base_config)
            config["C"] = C
            config["temp"] = temp
            
            # Save to a temporary config file
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Execute the sampling script with the temporary config
            subprocess.run([sys.executable, script_path, "--config", temp_config_path], check=True)
            print(f"========== Completed run for C={C}, temp={temp} ==========\n")
    # Cleanup the temporary file once all experiments are done
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

if __name__ == "__main__":
    run_batch()