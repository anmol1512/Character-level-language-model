import torch as tt
from torch.utils.data import Dataset
from model.utils.config import CfgNode as CN

'''This helps the DataLoader to pick sample based on an index'''
'''Map-style dataset'''
class TextDataset(Dataset):

    @staticmethod
    def get_default_config() -> CN:
        C=CN()
        C.block_size=256
        return C

    def __init__(self,config: CN,text_data: str) -> None:

        vocab = sorted(list(set(text_data)))

        self.vocab_size = len(vocab)
        self.config=config
        self.text_data = text_data
        self.stoi = {char:idx for idx,char in enumerate(vocab)}
        self.itos = {idx:char for idx,char in enumerate(vocab)}

    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def get_block_size(self) -> int:
        return self.config.block_size
    def __encoding(self,text_chunk: str) -> list:
        return [self.stoi[char] for char in text_chunk]

    def __len__(self) -> int:
        return len(self.text_data) - self.config.block_size
    
    def __getitem__(self,idx: int) -> tuple:

        # block_size referes to the length of input sequence we are working upon
        block_size=self.config.block_size
        # pluk block size contiguous sequence from the text data
        text_chunk=self.text_data[idx:idx+block_size+1]
        # encode each charater in the sequence 
        xy_chunk=self.__encoding(text_chunk)
        x=tt.tensor(xy_chunk[:-1], dtype=tt.long)
        y=tt.tensor(xy_chunk[1:], dtype=tt.long)

        return x,y
        
