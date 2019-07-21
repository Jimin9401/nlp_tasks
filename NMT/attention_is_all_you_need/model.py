import copy
import torch
import torch.nn as nn
import numpy as np
import math

import torch.nn.functional as F
from torch.autograd import Variable


class EncoderDecoder(nn.Module):

    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed

    def forward(self, src,tgt,src_mask,tgt_mask):

        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)

    def encode(self,src,src_mask):

        return self.encoder(self.src_embed(src),src_mask)

    def decode(self,memory,src_mask,tgt,tgt_mask):

        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)



class Generator(nn.Module):

    def __init__(self,d_model,vocab):

        super(Generator,self).__init__()
        self.proj=nn.Linear(d_model.vocab)


    def forward(self, x):

        return F.log_softmax(self.proj(x),dim=-1)

def clones(module,N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()

        self.layers=clones(layer,N)

        self_norm=LayerNorm(layer.size)

    def forward(self,input,mask):
        for layer in self.layers:
            input=layer(input,mask)


            return self.norm(input)



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


class SublayerConnection(nn.Module):

    def __init__(self,size,dropout):

        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x,sublayer):

        return x+self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,dropout),2)
        self.size=size

    def forward(self, x,mask):

        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))

        return self.sublayer[1](x,self.feed_forward)


class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self, x,memory,src_mask,tgt_mask):

        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)


        return self.norm(x)



class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,drop_out):
        super(DecoderLayer,self).__init__()

        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,drop_out),3)


    def forward(self,x,memory,src_mask,tgt_mask):

        m=memory

        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))

        return self.sublayer[2](x,self.feed_forward)




def subsequent_mask(size):
    attn_shape=(1,size,size)
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask)==0


def attention(query,key,value,mask=None,dropout=None):

    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1)) /math.sqrt(d_k)

    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)

    p_attn=F.softmax(scores,dim=-1)

    return torch.matmul(p_attn,value),p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):

        super(MultiHeadedAttention,self).__init__()

        assert d_model%h==0

        self.d_k=d_model//h
        self.h=h
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)


    def forward(self, query,key,value,mask=None):

        if mask is not None:
            mask=mask.unsqueeze(1)

        n_batches=query.size(0)


        query_key_value=\
            [l(x).view(n_batches,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]

        x,self_attn=attention(query,key,value,mask=mask,dropout=self.dropout)

        x=x.transpose(1,2).contiguous().view(n_batches,-1,self.h*self.d_k)

        return self.linears[-1](x)



class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):

        super(PositionwiseFeedForward,self).__init__()

        self.w_1=nn.linear(d_model,d_ff)
        self.w_2=nn.linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):

        super(Embeddings,self).__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model


    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):

        super(PositionwiseFeedForward,self).__init__()
        self.dropout=dropout

        pe=torch.zeros(max_len,d_model)

        position=torch.arange(0,max_len).unsqueeze(1)

        div_term=torch.exp(torch.arange(0,d_model,2)*(math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        self.register_buffer('pe',pe)

    def forward(self, x):
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)

        return self.dropout(x)



def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,dropout=0.1):

    c=copy.deepcopy()

    attn=MultiHeadedAttention(h,d_model)

    ff=PositionwiseFeedForward(d_model,d_ff,dropout)

    position=PositionalEncoding(d_model,dropout)

    model=EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
        Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),N),
        nn.Sequential(Embeddings(d_model,src_vocab),c(position)),
        nn.Sequential(Embeddings(d_model,tgt_vocab),c(position)),
        Generator(d_model,tgt_vocab))

    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)

    return model