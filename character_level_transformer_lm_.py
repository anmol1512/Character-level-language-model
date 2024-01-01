# -*- coding: utf-8 -*-
"""character-level Transformer LM .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uZ5sxia_jzwijlV5TXL6HGzjX2-hyZNo

## Single Head self-attention block
"""

import torch as tt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

batch_size= 32
block_size=256
learning_rate=1e-3
epochs=5000
eval_interval=100
eval_iter=200
n_embds=32
# head_size=16
max_token=500

tt.manual_seed(1512)

# Input dataset containing all the work of shakespeare
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Reading the file f-->input.txt
with open('input.txt','r') as f:
  input=f.read()

print("Total number of character in the file:",len(input))
# first 100 characters in the input file
print("preview of the input file:",end='\n\n')
print(input[:100])

# Defining vocabulary for creating a tokenizer
device = 'cuda' if tt.cuda.is_available() else 'cpu'
vocab=sorted(list(set(input)))
vocab_size=len(vocab)
print('vocab: '+''.join(vocab))
print(f'voacb size: {vocab_size}')
print(f'device: {device}')

# Our own character level tokenizer
ctoi={ele:ind for ind,ele in enumerate(vocab)}
itoc={ind:ele for ind,ele in enumerate(vocab)}
encode=lambda text:[ctoi[char] for char in text]
decode=lambda token_list:''.join([itoc[token] for token in token_list])


# Our charecter based tokenizer implementation/demo
e=encode("anmol is great")
d=decode(e)
print(e)
print(d)

# Encoded the input file into tokens of integer sequence
data=tt.tensor(encode(input),dtype=tt.long)
print(data.shape)
print(data[:10])
print(input[:10])

# Spliting the encode tensor data into training data and validation data
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

# Relation between context and target
block_size=8
x=train_data[:block_size]
y=train_data[1:block_size+1]
for i in range(block_size):
  context=x[:i+1]
  target=y[i]
  print(f'when input is {context} the target is: {target}')

# Data Loader Implementation
tt.manual_seed(1512)

def get_batch(data_type):
  data=train_data if data_type=='train' else val_data
  x_start=tt.randint(high=len(data)-block_size,size=(batch_size,))
  x= tt.stack([data[i:i+block_size] for i in x_start])
  y= tt.stack([data[i+1:i+block_size+1] for i in x_start])
  x=x.to(device)
  y=y.to(device)
  return x,y



xb,yb=get_batch('train')
print(xb.shape)
print(yb.shape)

# Custom implementation of Layer normalization layer by scratch
class LayerNorm:
  def __init__(self,dim,eps=1e-3):
    self.eps=eps
    self.gamma=tt.ones(dim,device=device)
    self.beta=tt.zeros(dim,device=device)
  def __call__(self,x):
    xmean=x.mean(-1,keepdim=True)
    xvar=x.var(-1,keepdim=True)
    xhat=(x-xmean)/(tt.sqrt(xvar+self.eps))
    out=xhat*self.gamma+self.beta
    return out
  def parameters(self):
    return {'gamma': self.gamma,'beta': self.beta}

class Head(nn.Module):
  def __init__(self,head_size):
    super().__init__()
    '''self-attention layers[query,key and value]'''
    self.query=nn.Linear(n_embds,head_size,bias=False)
    self.key=nn.Linear(n_embds,head_size,bias=False)
    self.value=nn.Linear(n_embds,head_size,bias=False)

    '''lower triangular matrix is registered as buffer for masking purpose'''
    self.register_buffer('lower_tri',tt.tril(tt.ones((block_size,block_size),dtype=tt.long))) # (T,T) tensor is created and registered as buffer.

  def forward(self,x): # x is the encoding of its identity+its position.
    B,T,C = x.shape
    q=self.query(x)
    k=self.key(x)
    wei=(q @ k.transpose(-2,-1)) * q.shape[-1]**-0.5#(B,T,T)
    '''wei is known as attention score as every token attends every other token,
     and then we normalize it and called them as scaled attention'''

    # -----DECODER-----
    wei=wei.masked_fill(self.lower_tri[:T,:T]==0,float('-inf')) #(B,T,T)
    wei=F.softmax(wei,dim=1) # Aggregating all the previous token information

    v=self.value(x)
    out=wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
    return out # input(B,T,n_embd) -> out(B,T,head_size)

class MultiHead(nn.Module):
  def __init__(self,num_heads,head_size):
    super().__init__()
    self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    # self.project=nn.Linear(num_heads*head_size,n_embds)
    self.project=nn.Linear(n_embds,n_embds)

  def forward(self,x):
    concat_heads=tt.cat([h(x) for h in self.heads],dim=-1)
    return self.project(concat_heads)

"""**Multi-Head self attention**

In the context of the multi-head self-attention mechanism in the Transformer architecture, the idea is to project the input into multiple subspaces (heads), each with its own learnable parameters. Each head computes a separate attention distribution, capturing different aspects of the relationships within the input sequence. The outputs from all heads are then concatenated and linearly transformed to produce the final output.

**Advantages**
1.  It acts as a form of regularization, making the model less sensitive to noise in the training data.
2. The attention weights produced by different heads can provide insights into which parts of the input sequence are relevant for specific heads.
3. Parallelization or parallel processing is computationally efficient in terms of usage of hardware resources and training time.
4. Reduce overfitting and makes model more robust and capable of generalizing well to different types of data
"""

class FeedForwardNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.ffnn=nn.Sequential(
        nn.Linear(n_embds,4*n_embds),
        nn.ReLU(),
        nn.Linear(4*n_embds,n_embds)
    )
  def forward(self,x):
    return self.ffnn(x)

""" **Feed forward neural network**
1. A feedforward neural network is a type of artificial neural network in which nodes' connections do not form a loop.
2. Often referred to as a multi-layered network of neurons.
3. feedforward neural networks are so named because all information flows in a forward manner only.
"""

class TransformerBlock(nn.Module):
  # communication followed by computation
  def __init__(self,n_heads,total_head_size):
    super().__init__()
    head_size=total_head_size//n_heads
    self.sa_heads=MultiHead(n_heads,head_size)
    self.neural_net=FeedForwardNN()
    self.ln1=nn.LayerNorm(n_embds)
    self.ln2=nn.LayerNorm(n_embds)

  def forward(self,x):
    x_attend=x+self.sa_heads(self.ln1(x))
    x_think=x_attend+self.neural_net(self.ln2(x_attend))
    return x_think

n_heads=4
n_decoder_layer=16
class nanogptmodel(nn.Module):
  def __init__(self):
    super().__init__()
    '''Encoding the input token sequence'''
    self.token_embedding=nn.Embedding(vocab_size,n_embds)
    self.position_embedding=nn.Embedding(block_size,n_embds)

    """
    '''Communication mechanism so that tokens can attend to its previous tokens.'''
    # self.sa_head=Head(n_embds)
    self.sa_heads=MultiHead(4, n_embds//4)

    '''Feed forward neural network block to think on the comunicated information operating on token level individually'''
    self.neural_net=FeedForwardNN()
    """
    # Sequential Transformer blocks
    self.block=nn.Sequential(*[TransformerBlock(n_heads,n_embds) for _ in range(n_decoder_layer)])
    self.ln_layer=nn.LayerNorm(n_embds)

    '''Decoding the information loaded encoded token sequence via transformation.'''
    self.lm_head=nn.Linear(n_embds,vocab_size)

  def forward(self,xb,yb=None):
    B,T=xb.shape

    token_embd=self.token_embedding(xb) #(B,T) -> (B,T,n_embds)
    position_embd=self.position_embedding(tt.arange(T,device=device)) #(T) -> (T,n_embds)
    x=token_embd+position_embd # (B,T,n_embds) ----[Broadcasting rules]

    '''
    x_attend=self.sa_heads(x) # (B,T,n_embds) -> (B,T,head_size)
    x_think=self.neural_net(x_attend) # (B,T,head_size) -> (B,T,n_embds)
    '''
    x=self.block(x) # Three transformer block in sequencial manner.
    x=self.ln_layer(x)

    logits=self.lm_head(x)  # (B,T,n_embds) -> (B,T,vocab_size)

    if yb==None:
      loss=None
    else:
      # xb -> (B,T)
      # yb -> (B,T)
      # yb'(logits) -> (B,T,vocab_size)
      B,T,C=logits.shape
      yb=yb.view(B*T)
      logits=logits.view(B*T,C)

      #negative log likelihood loss
      loss=F.cross_entropy(logits,yb)
    return logits,loss

  def generate(self,idx,max_token):

    for _ in range(max_token):

      idx_crop=idx[:,-block_size:]

      logits,_=self(idx_crop) #(B,T) -> (B,T,vocab_size)

      logits=logits[:,-1,:] #(B,T,vocab_size) -> (B,vocab_size)

      prob=F.softmax(logits,dim=-1) #(B,vocab_size) -> (B,vocab_size)

      next_idx=tt.multinomial(prob,num_samples=1) #(B,vocab_size) -> (B,1)

      idx=tt.cat((idx,next_idx),dim=-1) #(B,T) ->(B,T+1)

    return idx

model=nanogptmodel()
model=model.to(device)
logits,loss=model(xb,yb)
print(logits.shape)
print(loss)

context=tt.zeros((1,1), dtype=tt.long,device=device)
print(decode(model.generate(context,max_token)[0].tolist()))

@tt.no_grad()
def evaluate_loss():
  model.eval()
  losses={}
  for data_type in ('train','val'):
    loss_array=tt.zeros((eval_iter,))
    for i in range(eval_iter):
      xb,yb=get_batch(data_type)
      logits,loss=model(xb,yb)
      loss_array[i]=loss.item()
    losses[data_type+'_loss']=loss_array.mean().item()
  model.train()
  return losses

optimizer=tt.optim.AdamW(model.parameters(),lr=learning_rate)

for i in range(epochs):

  if i%eval_interval==0:
    loss=evaluate_loss()
    print(f"Step: {i}, train_loss: {loss['train_loss']:.4f}, val_loss: {loss['val_loss']:.4f}")

  #forward pass
  xb,yb=get_batch('train')
  logits,loss=model(xb,yb)

  #backpropagation
  optimizer.zero_grad(set_to_none=True)
  loss.backward()

  #update parameters
  optimizer.step()

print(decode(model.generate(context,max_token)[0].tolist()))

"""1. bigram model performance -> Train_loss: 2.4691, Val_loss: 2.4889.
2. single head self attention block performance -> train_loss: 2.1595, val_loss: 2.1689.
3. multi head self attention block performance -> train_loss: 0.6562, val_loss: 0.6810.
4. multi head self attention block +  Position-wise Feed-Forward Networks performance -> train_loss: 0.6378, val_loss: 0.6548.
5. 3 transformer block of point 4 -> train_loss: 0.3761, val_loss: 0.3830.

6. 3 transformer block of point 4 with skip connection in each block -> train_loss: 0.3154, val_loss: 0.3265, NOTE: point 5 losses were reached in approx epochs/2 training time.

7. Added projection layer for both multihead and FFNN and added 1 hidden layer in FFNN -> train_loss: 0.2998, val_loss: 0.3077.
"""