import torch as tt
from torch.utils.data import Dataset
from utils.config import CfgNode as CN


class TextDataset(Dataset):

    '''
    Map-style dataset

    This helps the DataLoader to pick sample based on an index
    '''
    @staticmethod
    def get_default_config() -> CN:
        C=CN()
        C.min_len = 5
        C.block_size=256
        C.train_file_path = '/data/WMT2014_en-hi/train'
        C.test_file_path =  '/data/WMT2014_en-hi/test'
        C.val_file_path =  '/data/WMT2014_en-hi/val'
        return C

    def __init__(self,config: CN,x: str, y: str) -> None:
        self.config=config
        self.x = x
        self.y = y
        self.sequence_len = config.block_size

        # data corpus
        corpus = ''.join(x+y)
        vocab = ['<pad>'] +['<sos>'] + ['<eos>'] + sorted(list(set(corpus)))
        data_size, vocab_size = len(corpus), len(vocab)

        #data specs
        print(f'Data has {data_size} total characters')
        print(f'Data has {vocab_size-3} total unique chars')
        print(f'Data has {len(x)} total sentences')

        self.vocab_size = vocab_size    
        self.ctoi = {char:idx for idx,char in enumerate(vocab)}
        self.itoc = {idx:char for idx,char in enumerate(vocab)}

    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def get_block_size(self) -> int:
        return self.config.block_size
    
    def encoding(self,text_chunk: str) -> list[int]:
        return [self.ctoi[char] for char in text_chunk]

    def __len__(self) -> int:
        '''Total samples in dataset'''
        return len(self.x)
    
    def __getitem__(self,idx: int) -> tuple[tt.tensor[int],tt.tensor[int]]:
        '''
        input to the encode is of the format  `This is a car<EOS>`

        output[logits] of the model is of the format `ye ek car hai<EOS>`

        input to the decoder is of the format `<SOS>ye ek car hai`

        we then compare `ouput` of the model with `input of decoder[1:]` to calculate the nll and BLUE Score

        block_size referes to the length of input sequence we are working upon

        '''
        start=[self.ctoi['<sos>']]
        end=[self.ctoi['<eos>']]
        
        indx = self.padding(self.encoding(self.x[idx]) + end)
        indy = start + self.padding(self.encoding(self.y[idx]) + end)

        x = tt.tensor(indx, dtype=tt.long)
        y = tt.tensor(indy, dtype=tt.long)

        return x,y
    
    def padding(self, seq: list[int]) ->list[int]:
        seq_len = len(seq)
        if seq_len < self.sequence_len:
            seq =  seq + [0]*(self.sequence_len - seq_len)
        else: 
            seq =seq[:self.sequence_len-1] + [self.ctoi['<eos>']]
        return seq
