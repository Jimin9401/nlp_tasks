from utils import dataloader
import torch

import torch.nn as nn

import torch.nn.functional as F

train,val,test,vocab=dataloader(batch_size=32,memory_size=5,task=1,joint=False,tenK=False)



train

for batch in train:
    print()
    memory=batch.story
    query=batch.query
    answer=batch.answer

len(vocab)

embedder=nn.Embedding(num_embeddings=len(vocab.stoi),embedding_dim=300)

memory_embed=embedder.forward(memory)
query_embed=embedder.forward(query)


query_embed.size()




m=torch.sum(memory_embed,dim=2)/memory_embed.size(2)

m.size()

p=torch.matmul(query_embed,m.transpose(-1,-2))/np.sqrt(300)

attn_score=F.softmax(p,dim=2)

query_embed.size()

o=torch.matmul(attn_score,m)

