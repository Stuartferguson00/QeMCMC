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
    times = np.logspace(-1,1, 5).tolist()
    h_s = np.arange(0.1, 1, 0.2).tolist()
    print("times: ", times)
    print("h_s: ", h_s)
    C = 20
    temp = 1000
    uniform = True


    config = copy.deepcopy(base_config) 
    config["experiments"] = []
    index= 0
    for t in times:
        for h in h_s:
            print(f"========== Starting run for C={C}, h={h} ==========")
            
            # Create a copy of the base config and modify the parameters
            print("t: ", t)
            print("h: ", h)
            config["C"] = C
            config["temp"] = temp
            config["uniform"] = uniform
            # set the experiment parameters of C and temp
            config["initial_states"] =  [
                                            "0",
                                            
                                        ]
            config["experiments"].append({
                    "name": "qcst_gamma"+str(index),
                    "type": "qe",
                    "params": {
                        "gamma": h,
                        "time": t,
                        "m": 19,
                        "repeated": False
                    }
                    })
            index += 1

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