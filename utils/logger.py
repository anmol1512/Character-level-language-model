import sys
import os
import yaml

def setup_logging(config):
    # working directory specified in default_config
    work_dir=config.system.config_work_dir
    #created a directory where we will log our configurations
    os.makedirs(work_dir,exist_ok=True) 
    
    # write the current command line argument in cli_args.txt file
    with open(os.path.join(work_dir+'cli_args.txt'),'w') as f:
        f.write(' '.sys.argv)
    
    #create and write the current configuration in a yaml file config.yaml
    with open(os.path.join(work_dir+'config.yaml'),'w') as f:
        yaml.dump(config._dict_repr(),f,default_flow_style=False)


