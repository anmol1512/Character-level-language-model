# -*- coding: utf-8 -*-
"""Character-based_bigram_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15n8_TZKiFAuxRxasjrSMdctZCFFaoLBl

## Character-based Transformer language model
"""
# --------------importing all library and modules used-----------------
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import time
# --------------importing all library and modules used-----------------

# ----------------------Hyperparameters------------------
batch_size=32
block_size=8
epochs=10000
eval_interval=100
eval_iter=200
learning_rate=1e-3
# ----------------------Hyperparameters------------------

t.manual_seed(1512)

# --------------Reading the file f-->input.txt-----------------

# Input dataset containing all the work of shakespeare(Dataset name -Tiny shakespeare )
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt','r') as f:
  input=f.read()

# first 100 characters in the input file
print("preview of the input file:",end='\n\n')
print(input[:100])
print("Total number of character in the file:",len(input))

# --------------Reading the file f-->input.txt-----------------

# --------------Defining vocabulary for creating a tokenizer---------

device = 'cuda' if t.cuda.is_available() else 'cpu' #selection of device for computations
vocab=sorted(set(input))
vocab_size=len(vocab)
print('vocab: '+''.join(vocab))
print(f'voacb size: {vocab_size}')
print(f'device: {device}') 

# --------------Defining vocabulary for creating a tokenizer---------

# -----------our own character level tokenizer-------------------

ctoi={ele:ind for ind,ele in enumerate(vocab)}
itoc={ind:ele for ind,ele in enumerate(vocab)}
encode=lambda text:[ctoi[char] for char in text]
decode=lambda token_list:''.join([itoc[token] for token in token_list])

# our charecter based tokenizer implementation/demo
e=encode("anmol is great")
d=decode(e)
print(e)
print(d)

# -----------our own character level tokenizer-------------------


'''
# tiktoken is a OpenAI's BPE based tokenizer for subword level encoding of text to integer based on a vocabulary
# !pip install tiktoken
# import tiktoken as tt
# tokenizer= tt.get_encoding('gpt2')
# print(tokenizer.n_vocab)
# print(tokenizer.encode("anmol is great"))
'''

# ------------Tokeninzing the whole dataset--------------
data=t.tensor(encode(input),dtype=t.long)
print(data.shape)
print(data[:10])
print(input[:10])
# ------------Tokeninzing the whole dataset--------------

# ----------------Spliting into training data and validation data------------
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]
# ----------------Spliting into training data and validation data------------


x=train_data[:block_size]
y=train_data[1:block_size+1]
for i in range(block_size):
  context=x[:i+1]
  target=y[i]
  print(f'when input is {context} the target is: {target}')


# ------------------DataLoader---------------------------
def get_batch(data_type):
  data=train_data if data_type=='train' else val_data
  x_start=t.randint(high=len(data)-block_size,size=(batch_size,))
  x= t.stack([data[i:i+block_size] for i in x_start])
  y= t.stack([data[i+1:i+block_size+1] for i in x_start])
  x=x.to(device)
  y=y.to(device)
  return x,y
# ------------------DataLoader---------------------------

xb,yb=get_batch('train')
print(xb.shape)
print(yb.shape)

'''
# for i in range(batch_size):
#   for j in range(block_size):
#     context=xb[i,:j+1]
#     target=yb[i,j]
#     print(f"when the input is {context.tolist()} the target is: {target})")
'''
# --------------------------Evaluate training and validation loss---------------------
@t.no_grad()
def evaluate_loss():
  losses={}
  model.eval()
  for data_type in ['train','val']:
    loss_array=t.zeros((eval_iter))
    for i in range(eval_iter):
      xb,yb=get_batch(data_type) # Data Loader
      logits,loss=model(xb,yb)
      loss_array[i]=loss.item()
    losses[data_type+'_loss']=loss_array.mean().item()
  model.train()
  return losses
# --------------------------Evaluate training and validation loss---------------------

# -------------------------Bigram model definition------------------------
class Bigrammodel(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.lookup=nn.Embedding(vocab_size,vocab_size)

  def forward(self,xb,yb=None):
    logits=self.lookup(xb)#(B,T,C)->(4,8,65)
    if yb==None:
      return logits,None
    B,T,C=logits.shape
    logits=logits.view(B*T,C)#(32,65)
    yb=yb.view(B*T)#(32,)

    loss=F.cross_entropy(logits,yb)
    return logits,loss

  def generate(self,xb,max_new_tokens):
    for i in range(max_new_tokens):
      logits,_=self(xb)#(4,8)->(4,8,65)
      logits=logits[:,-1,:]#(4,65)

      prob=F.softmax(logits,dim=1)#(4,64)
      next_idx=t.multinomial(prob,num_samples=1)#(4,1)

      xb=t.cat((xb,next_idx),dim=1)#(4,9)
    return xb
# -------------------------Bigram model definition------------------------

# -------------------------Bigram model without training------------------
model=Bigrammodel(vocab_size)
model=model.to(device)
logits,loss=model(xb,yb)
print(logits.shape)
print(f'loss: {loss.item()}')

context=t.zeros((1,1),dtype=t.long,device=device)
max_new_tokens=100
# generate text from model before training
print(decode(model.generate(context,max_new_tokens)[0].tolist()))
# -------------------------Bigram model without training------------------


# -------------------------Bigram model with training------------------
# pytorch optimizer AdamW
optimizer=t.optim.AdamW(model.parameters(),lr=learning_rate)
start_time=time.time()

for i in range(epochs):
  if i%eval_interval==0:
    losses=evaluate_loss()
    print(f"Step: {i+1}, Train_loss: {losses['train_loss']:.4f}, Val_loss: {losses['val_loss']:.4f}")
  # get input
  xb,yb=get_batch('train')

  # forward pass
  logits,loss=model(xb,yb)
  optimizer.zero_grad(set_to_none=True)

  # backward pass
  loss.backward() # calculate gradient w.r.t all the model parameters
  # update the model parameter
  optimizer.step()

end_time=time.time()
print(f'Total time taken for training in {device} is: {end_time-start_time}')
# Total time taken for training in cpu is: 87.45440864562988 sec
# Total time taken for training in cuda is: 71.0542266368866 sec

# generate text from model after training
print(decode(model.generate(context,max_new_tokens)[0].tolist()))
# -------------------------Bigram model with training------------------
