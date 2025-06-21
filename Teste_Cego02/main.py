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

    # Explorer needs to know rescuer to send the map 
    # that's why rescuer is instatiated before

    # Movement priority vectors per explorer (used to diversify exploration directions)
    vectors = {
        1: [2, 1, 0, 7, 6, 5, 4, 3],
        2: [2, 3, 4, 5, 6, 7, 0, 1],
        3: [6, 7, 0, 1, 2, 3, 4, 5],
        4: [6, 5, 4, 3, 2, 1, 0, 7]
    }

    # Instantiate each explorer with its vector
    for exp in range(1, 5):
        explorer_file = os.path.join(config_ag_folder, f"explorer_{exp}_config.txt")
        Explorer(env, explorer_file, master_rescuer, vectors[exp])
        
        # Explorer(env, explorer_file, master_rescuer)

    # Run the environment simulator
    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_400v_90x90")
        config_ag_folder_name = os.path.join("", "cfg_1")
        
    main(data_folder_name, config_ag_folder_name)