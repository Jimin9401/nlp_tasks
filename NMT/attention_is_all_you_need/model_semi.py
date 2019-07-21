import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import copy
import numpy as np



def clones(module, N):
    layers = nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    return layers

class Embedding(nn.Module):

    def __init__(self,d_model,vocab_size,padding_idx=0):
        super(Embedding,self).__init__()
        self.embedding=nn.Embedding(padding_idx=padding_idx,num_embeddings=vocab_size,embedding_dim=d_model)

    def forward(self, x):

        # x = [ batch , seq_lens]
        x_embed=self.embedding(x)
        # x= [batch,seq_lens, d_model]


        return x_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, H,padding_idx):
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model;
        self.H = H
        self.d_k = d_model // H
        self.linear = nn.Linear(d_model, d_model)
        self.linears = nn.ModuleList([copy.deepcopy(self.linear) for _ in range(4)])
        self.padding_idx=padding_idx

    def forward(self, query,key,value,mask):
        # input= [batch, seq_lens,d_model]
        n_batch = query.size(0)
        mask=mask.unsqueeze(1)

        # q,k,v=[batch, seq_lens, head, d_k]
        query, key, value = [L.forward(x).view(n_batch, -1, self.H, self.d_k).transpose(1, 2) for L, x in
                             zip(self.linears, (query, key, value))]

        w=torch.matmul(query, key.transpose(-2, -1))/np.sqrt(self.d_k)

        w=w.masked_fill(mask==0,-1e9)


        attn_score = F.softmax(w, dim=-1)

        z = torch.matmul(attn_score, value)
        z = z.view(n_batch, -1, self.d_model)

        # z=[batch,seq_lens,d_model] :: concatenate multihead

        return self.linears[-1].forward(z)



class PositionwiseFeedforward(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super(PositionwiseFeedforward,self).__init__()

        self.linear1=nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout=nn.Dropout(dropout)

    def forward(self, x):

        return self.linear2(F.relu(self.linear1(x)))


class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()

        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))

        self.eps=eps

    def forward(self, x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)

        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

class SubLayerConnection(nn.Module):
    def __init__(self,size):
        super(SubLayerConnection,self).__init__()
        self.normalize=LayerNorm(features=size)

    def forward(self, x,sublayer):

        return x+sublayer(self.normalize(x))



class Encoder(nn.Module):
    def __init__(self,N,sublayer):
        super(Encoder).__init__()
        self.N=N
        self.layers=clones(sublayer,N)

    def forward(self, input,mask):

        for layer in self.layers:
            input=layer.forward(input,mask)

        return input


class EncoderLayers(nn.Module):
    def __init__(self,N,sublayer,attention,feed_forward):
        super(EncoderLayers,self).__init__()
        self.N=N
        self.sublayers=clones(sublayer,N) ## s

        self.attention=attention  ## scale dot product mechanism
        self.feed_forward=feed_forward ## Positionwise FF network

    def forward(self, x,mask):

        x=self.sublayers[0](x, lambda x:self.attention(x,x,x,mask))

        x=self.sublayers[1](x,self.feed_forward)

        return x



