from collections import defaultdict
from utils.config import CfgNode as CN
import torch as tt
from torch.utils.data import DataLoader,RandomSampler
from model.transformer import NanoGPTModel as nano
from data.dataset import TextDataset
import time
import tqdm

class Trainer:

    @staticmethod
    def get_default_config():
        C=CN()
        C.checkpoint_work_dir = 'checkpoint'

        #data loader parameter
        C.num_workers = 2

        # optimizer Hyperparameters
        C.max_epoch = None
        C.learning_rate = 1e-4
        C.weight_decay = 0.1
        C.betas = (0.9,0.98)
        C.batch_size = 128

        #inference config
        C.eval_interval = 500
        C.eval_iter = 300
        C.max_token = 2000
        return C

    def __init__(self,config: CN,model: nano,train_data: TextDataset,val_data: TextDataset,test_data: TextDataset) -> None:
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = model.configure_optimizer(self.config)
        self.callbacks = defaultdict(list)
        self.device = model.device
        
        print('SENDING MODEL TO {}.....................'.format(self.device))
        self.model=self.model.to(self.device)
        print('\nSuccess!!')

        
        self.current_epoch=0


    '''event could be a user requested for a search query and,
      depending on the type of seaerch query it can have many callbacks
    #example onevent seach query there are two callbacks(function that perform search) 
    one refers to search by name and another callback refers search by age'''

    '''adding callback(function) on an event means 
    that wherver an event is occured we need to add an function which can be triggered'''
    def add_callback(self,callback,event) -> None:
        # self.callbacks[event].append(callback) if not self.callbacks.get(event,[]) else self.callbacks[event].update(dict(event=[callback]))
        self.callbacks[event].append(callback)
    
    '''we are setting up another callback(function) which can be triggered 
    when an event occured'''
    def set_callback(self,callback,event) -> None:
        self.callbacks[event]=[callback]

    def trigger_callbacks(self,event) -> None:
        callbacks=self.callbacks[event]
        for callback in callbacks:
            callback(self)


    def run(self):
        #for code optimization
        train_data=self.train_data
        model=self.model
        train_config=self.config
        max_epoch=train_config.max_epoch
        current_epoch=self.current_epoch
        optimizer=self.optimizer
        
        min_loss = float('inf')
        best_model_state = None
        best_optimizer_state = None
        best_step = None
        for curr_epoch in range(max_epoch):
            model.train()
            sampler = RandomSampler(train_data,replacement=False)
            train_batches = DataLoader(
                train_data,
                batch_size = train_config.batch_size,
                sampler = sampler,
                shuffle = True,
                pin_memory = True,
                num_workers = train_config.num_workers
                )
            
            prog_bar = tqdm(enumerate(train_batches), total = len(train_batches))
            for idx, (x,y) in prog_bar:
                '''GET A BATCH'''
                x = x.to(self.device)
                y = y.to(self.device)
                
                ''' FORWARD PASS & COMPUTE LOSS'''
                _,loss = model(x,y)
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    best_step = idx
                    best_model_state = model.state_dict()
                    best_optimizer_state = optimizer.state_dict()

                '''BACK PROPAGATION'''
                optimizer.zero_grad(set_to_none=True)
                loss.backward() #gradients computed

                '''UPDATE PARAMETERS'''
                optimizer.step()
                
                            
            checkpoint_data = dict(
                    epoch = curr_epoch,
                    step = best_step,
                    model_state = best_model_state,
                    optim_state = best_optimizer_state
                )
            tt.save(checkpoint_data,train_config.checkpoint_work_dir+'/checkpoint.pt')
            self.trigger_callbacks('batch_end')

            
            
        

    

