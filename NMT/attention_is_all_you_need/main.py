from load_csv import load_pair
from preprocessing import tokenize,make_dictionary,convert2idx
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import numpy as np
from utils import pad_sequence,makeData
import copy



from model_semi import Embedding,clones,LayerNorm,PositionwiseFeedforward,SubLayerConnection,EncoderLayers,Encoder,MultiHeadAttention

import torch.nn.functional as F

trg,src=load_pair("../da    ")


src_ix=tokenize(src)

tgt_ix=tokenize(trg)

eng2idx,idx2eng=make_dictionary(src_ix)
tgt2idx,idx2es=make_dictionary(tgt_ix)


src_sentences=convert2idx(src_ix,eng2idx)

tgt_sentences=convert2idx(tgt_ix,tgt2idx)



train_data=makeData(src_sentences,tgt_sentences)


train_loader=DataLoader(dataset=train_data,collate_fn=pad_sequence,batch_size=100)

for (X_batch,y_batch) in train_loader:
    print(X_batch)



embedding=Embedding(vocab_size=len(eng2idx),d_model=512,padding_idx=0)

X_embed=embedding(X_batch)

X_pad=(X_batch==0).unsqueeze(-2)

norm=LayerNorm(512)

sublayer=SubLayerConnection(512)
attn=MultiHeadAttention(512,8,0)

ff=PositionwiseFeedforward(512,2048,0.1)

Encoders=EncoderLayers(2,sublayer=sublayer,attention=attn,feed_forward=ff)

Encoders.forward(X_embed,X_pad).size()
# encoder 6-stacks

#remain.. decoder part

