from model.transformer import NanoGPTModel as nano
from data.dataset import TextDataset
from model.trainer import Trainer
from utils.config import CfgNode as CN
from utils.logger import setup_logging
from utils.callbacks import evaluate_loss
from utils.pre_processing import load_data
import torch as tt
import sys
import os

# Getting all config
def get_config():
    C = CN()

    # SYSTEM CONFIG
    C.system = CN()
    C.system.manual_seed = 1902
    C.system.config_work_dir = 'config'

    # MODEL CONFIG
    C.model = nano.get_default_config()
    C.model.n_block = 8
    C.model.n_heads = 16
    C.model.n_embds = 512
    
    #DATA CONFIG
    C.data = TextDataset.get_default_config()

    # TRAINER CONFIG
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate=1e-4

    return C



if __name__ == '__main__':

    # Deal with config
    config = get_config()
    user_config = sys.argv[1:]
    config.update_args(user_config)

    #construct the dataset
    print('LOADING DATA.........................')
    '''load dataset and create a custom dataset i.e. TextDataset''' 
    train_x,train_y = load_data(x_path = config.data.train_file_path+'/en',y_path = config.data.train_file_path+'/hi',min_len = config.data.min_len,max_len = config.data.block_size)
    train_data = TextDataset(config.data,train_x,train_y) # sequence_length is directly propotional to BLUE score
    

    #construct the model
    print('SETTING UP MODEL.....................')
    '''create transformer model'''
    config.model.vocab_size = train_data.get_vocab_size()
    config.model.block_size = train_data.get_block_size()
    model = nano(config.model)
    
    '''log your data `datetime, cli args via sys.argv and model parameters`'''
    print('LOGGING YOUR INFO.....................')
    setup_logging(config,model.get_num_parameters())

    #construct the trainer
    '''Customized training loop with a callback to evaluate loss'''
    trainer = Trainer(config.trainer, model, train_data)
    trainer.set_callback(evaluate_loss,'batch_end')

    '''Loading pre-trained weights'''
    try:
        checkpoint_path = os.join.path(trainer.checkpoint_work_dir + '/checkpoint.pt')
        checkpoint_dict = tt.load(checkpoint_path)
        model.load_state_dict(checkpoint_dict['model_state']) # load model state
        #load the optimizer also.......       
    finally:
        trainer.run()


    