
# batch_begin_callback() is trigger wat the beginning of every epoch
'''It runs a loop on each data batch and takes 'len(dataloader)' steps to find the global minima'''
def batch_begin_callback(trainer_confifg):
    pass




#after the updation on each batch, batch_end_callback() is triggered
'''1. we evaluate train_loss and val_loss
2. save model checkpoint if current epoch loss is less than min_loss
3. if 2 happens we also save the generated text from model and 
4. if 2 happens evaluate and save the BLUE score'''
def batch_end_callback(trainer_config):
    pass