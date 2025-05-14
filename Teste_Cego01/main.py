import sys
import os
import time

## importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer

def main(data_folder_name, config_ag_folder_name):
   
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
    
    # Instantiate the environment
    env = Env(data_folder)
    
    # Instantiate master_rescuer
    # This agent unifies the maps and instantiate other 3 agents
    rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, 4)   # 4 is the number of explorer agents

    #Vetor de prioridade de movimentos
    # v1 = [0, 1, 2, 3, 4, 5, 6, 7]  # Prioritize moving in a clockwise direction
    # v2 = [0, 2, 4, 6, 1, 3, 5, 7]  # Prioritize moving in a zigzag pattern
    # v3 = [7, 6, 5, 4, 3, 2, 1, 0]  # Prioritize moving in a counter-clockwise direction
    # v4 = [7, 5, 3, 1, 6, 4, 2, 0]  # Prioritize moving in a reverse zigzag pattern

    v1 = [2, 1, 0, 7, 6, 5, 4, 3]
    v2 = [2, 3, 4, 5, 6, 7, 0, 1]
    v3 = [6, 7, 0, 1, 2, 3, 4, 5]
    v4 = [6, 5, 4, 3, 2, 1, 0, 7]

    # Explorer needs to know rescuer to send the map 
    # that's why rescuer is instatiated before
    for exp in range(1, 5):
        filename = f"explorer_{exp:1d}_config.txt"
        explorer_file = os.path.join(config_ag_folder, filename)
        # Add vector of priorities to the explorer
        if exp == 1:
            Explorer(env, explorer_file, master_rescuer, v1)
        elif exp == 2:
            Explorer(env, explorer_file, master_rescuer, v2)
        elif exp == 3:
            Explorer(env, explorer_file, master_rescuer, v3)
        else:
            Explorer(env, explorer_file, master_rescuer, v4)

    # Run the environment simulator
    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_225v_100x80")
        config_ag_folder_name = os.path.join("cfg_1")
        
    main(data_folder_name, config_ag_folder_name)
