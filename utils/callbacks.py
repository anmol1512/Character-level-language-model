import torch as tt
from torch.utils.data import DataLoader,RandomSampler
from model.trainer import Trainer

# After the updation on each batch, batch_end_callback() is triggered
@tt.no_grad()
def evaluate_loss(trainer: Trainer) -> None:
    '''
    Evaluate train_loss and val_loss
    '''
    trainer.model.eval()
    train_loader = DataLoader(
        trainer.train_data,
        batch_size = trainer.config.batch_size,
        sampler = RandomSampler(trainer.train_data,replacement=False),
        shuffle = False,
        pin_memory = True,
        num_workers = trainer.config.num_workers
        )
    val_loader = DataLoader(
        trainer.val_data,
        batch_size = trainer.config.batch_size,
        sampler = RandomSampler(trainer.val_data,replacement=False),
        shuffle = False,
        pin_memory = True,
        num_workers = trainer.config.num_workers
        )
    
    losses={}
    eval_iter = min([len(train_loader),len(val_loader)])
    for loss_type in ('val_loss','train_loss'):
        loss_array=tt.zeros((eval_iter,))
        batches = train_loader if loss_type == 'train_loss' else val_loader
        for idx,(x,y) in enumerate(batches):
            if idx == eval_iter:
                break
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            _,loss = trainer.model(x,y)
            loss_array[idx] = loss.item()
        losses[loss_type]=loss_array.mean().item()
    print(end = ' ')
    print(f'Train_loss: {losses["train_loss"]} | Val_loss: {losses["val_loss"]}')