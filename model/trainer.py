from collections import defaultdict
from model.utils.config import CfgNode as CN
import torch as tt
from torch.utils.data import Dataloader,RandomSampler
import time

class Trainer:

    @staticmethod
    def get_default_config():
        C=CN()

        #device type
        C.device='auto'

        #data loader parameter
        C.num_workers=2

        # optimizer Hyperparameters
        C.max_epoch=None
        C.learning_rate=1e-4
        C.weight_decay=0.1
        C.betas=(0.9,0.95)
        C.batch_size=128

        #inference config
        C.eval_interval=500
        C.eval_iter=300
        C.max_token=2000

    def __init__(self,config,model,train_data) -> None:
        self.config=config
        self.model=model
        self.train_data=train_data
        self.optimizer=model.configure_optimizer(self.config)
        self.callbacks=defaultdict(list)

        if config.device=='auto':
            self.device=tt.device('cuda:0') if tt.cuda.is_available() else tt.device('cpu')
        else:
            self.device=config.device
        
        print('.....Sending model to {}......'.format(self.device))
        self.model=self.model.to(self.device)
        print('.....Running on {}.......'.format(self.device))

        
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
            callback()
    
    def run(self):
        #for code optimization
        train_data=self.train_data
        model=self.model
        train_config=self.config
        max_epoch=train_config.max_epoch
        current_epoch=self.current_epoch

        sampler = RandomSampler(train_data,replacement=False)
        train_batches = Dataloader(
            train_data,
            batch_size=train_config.batch_size,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
            num_workers=train_config.num_workers
            )
        

        model.train()
        it=iter(train_batches)
        ''' Note: 1 EPOCH IS 1 STEP TOWARDS GLOBAL OPTIMA POINT'''
        while True:
            start_time = time.time()

            # GET A BATCH OF DATA
            try:
                batch = it.__next__()
            except StopIteration:
                it = iter(train_batches)
                batch = it.__next__()
            batch = [tup.to(train_config.device) for tup in batch]
            xb,yb = batch

            # FORWARD PASS
            logits,loss = model(xb,yb)

            # BACK PROPAGATION
            tt.zero_grad(set_to_none=True)
            loss.backward() #gradients computed

            # UPDATE PARAMETERS
            self.optimizer.step()



            end_time=time.time()
            step_time=end_time-start_time
            self.trigger_callbacks('batch_end')
            current_epoch+=1


            if max_epoch is not None and current_epoch>=max_epoch:
                break

            
            
        

    

