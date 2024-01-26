import sys
import os
import yaml
from datetime import datetime

def setup_logging(config,model_total_parameter):
    # working directory specified in default_config
    work_dir=config.system.config_work_dir
    #created a directory where we will log our configurations
    os.makedirs(work_dir,exist_ok=True) 
    
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

     # Append mode for cli_args.txt file
    with open(os.path.join(work_dir+'/cli_args.txt'), 'a') as f:
        # Save timestamp, number of parameters, and command line arguments
        f.write(f'{timestamp} - Model parameters: {model_total_parameter/1e6:.2f}M - {" ".join(sys.argv)}\n')
    
    #create and write the current configuration in a yaml file config.yaml
    with open(os.path.join(work_dir+'/config.yaml'),'w') as f:
        yaml.dump(config.dict_repr(),f,default_flow_style=False)


