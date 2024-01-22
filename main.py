from model.transformer import NanoGPTModel as nano
from data.dataset import TextDataset
from model.trainer import Trainer
from utils.config import CfgNode as CN
from utils.logger import setup_logging
from utils.callbacks import batch_begin_callback
from utils.callbacks import batch_end_callback
from utils.pre_processing import load_data
import sys

# Getting all config
def get_config():
    C = CN()

    # SYSTEM CONFIG
    C.system = CN()
    C.system.manual_seed = 1902
    C.system.config_work_dir = '/config'
    C.system.checkpoint_work_dir = '/checkpoint'

    # MODEL CONFIG
    C.model = nano.get_default_config()
    C.model.n_decoder_layer = 8
    C.model.n_heads = 8
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
    config.update_arg(user_config)
    print('**************************CONFIG**************************\n')
    print(config)
    print('**************************CONFIG**************************\n')

    #construct the dataset
    '''load dataset and create a TextDataset''' 
    train_x,train_y = load_data(x = config.data.train_file_path+'/en',y = config.data.train_file_path+'/hi',min_len = config.data.min_len,max_len = config.data.block_size)
    train_data = TextDataset(config.data,train_x,train_y) # sequence_length is directly propotional to BLUE score
        

    #construct the model
    model = nano(config.model)

    setup_logging(config,model.get_num_parameters())

    #construct the trainer
    trainer = Trainer(config.trainer, model, dataset)

    trainer.set_callback(batch_begin_callback,'batch_begin')
    trainer.set_callback(batch_end_callback,'batch_end')

    trainer.run()


    