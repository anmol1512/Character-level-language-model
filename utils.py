import sys
import os
from ast import literal_eval
import pyyaml
import json



def setup_logging(config):
    # working directory specified in default_config
    work_dir=config.system.work_dir
    #created a directory where we will log our configurations
    os.makedirs(work_dir,exist_ok=True) 
    
    # write the current command line argument in cli_args.txt file
    with open(os.path.join(work_dir+'cli_args.txt'),'w') as f:
        f.write(' '.sys.argv)
    
    #create and write the current configuration in a yaml file config.yaml
    with open(os.path.join(work_dir+'config.yaml'),'w') as f:
        yaml.dump(config._dict_repr(),f,default_flow_style=False)



''' A lightweight configuration system inspired by yacs[yet another configuration system] '''
class CfgNode:
    #Below are the target of this class
    # 1. Able to represent current configuration in a human redable block style manner like in yaml file config
    # 2. Able to represent current configuration in a nested dictionary format similar to yaml.load()
    # 3. Able to ovewrite the configuration using command line.
    def __init__(self,**kwagrs):
        vars(self).update(kwagrs)
    
    def __str__(self):
        return self._block_repr(0)
    
    def _block_repr(self,indent):
        line=[] # contains each line of a yaml file
        for key,val in vars(self).items():
            if isinstance(val,CfgNode):
                line.append('{}:\n'.format(key))
                line.append(val._block_repr(indent+1))
            else:
                line.append('{}: {}\n'.format(key,val))
            
        line=[' '*indent+l for l in line]
        return ''.join(line)
    
    def _dict_repr(self):
        return {key: val._dict_repr() if isinstance(val,CfgNode) else val for key,val in vars(self).items()}
    

    def _update_args(self,args):
        '''Override an existing config attribute by getting input
        from sys.argv[1:]'''

        '''The input is of the format --arg=value,
        where arg can be nested form attribute. 
        Hence we need to handle the nested case also.'''
        
        # Note: The (.) in arg is used to denote nested  sub-attributes.
        '''eg- model.attention.drop_ratio=0.5'''

        for arg in args:
            pair=arg.split('=')
            assert len(pair)==2, 'Expected override value is missing. Got {} instead'.format(arg)

            key,val=pair
            #Handling value
            try:
                val=literal_eval(val)

            except ValueError:
                exit()

            #Handling the key
            assert key[:2]=='--', 'Config attriute format is illegal. Got {} instead'.format(key)
            keys=key[2:]
            keys=keys.split('.')

            obj=self
            for attr in keys[:-1]:
                obj=getattr(obj,attr)
            
            leaf_attr=keys[-1]
            assert hasattr(obj,leaf_attr),  '{} attribute Not Found in the config setting!!'.format(key[2:])
            
            print(f'Overwritting attribute {key[2:]} from {getattr(obj,leaf_attr)} to {val} in config setting.....')
            setattr(obj,leaf_attr,val)



            








