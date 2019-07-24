import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from eval import greedy_decode

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def attention(query, memory, context, mask=None, dropout=None):
    scores = torch.matmul(query, memory.transpose(1, 2)) / np.sqrt(memory.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

    p_attn = F.softmax(scores, dim=-2)

    return torch.matmul(p_attn, context)


def position_encoding_init(n_position, emb_dim):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor).to(device)



class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a=torch.ones(features)
        self.b=torch.zeros(features)
        self.eps=eps
    def forward(self, x):

        m=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)

        return (self.a*(x-m)/std+self.eps)+self.b


class Memory(nn.Module):
    def __init__(self, vocab_size, d):
        super(Memory, self).__init__()
        self.embedder_A = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)
        self.embedder_C = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)

    def forward(self, sentence):
        m = self.embedder_A.forward(sentence)
        c = self.embedder_C.forward(sentence)

        # [n_batch, memory_size, seq_lens, vector_dim]

        seq_lens = m.size(2)
        dim = m.size(-1)

        m += position_encoding_init(seq_lens, dim)
        c += position_encoding_init(seq_lens, dim)

        return torch.sum(m, dim=-2), torch.sum(c, dim=-2)


class Query(nn.Module):
    def __init__(self, vocab_size, d):
        super(Query, self).__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)

    def forward(self, query):

        return torch.sum(self.embedder.forward(query),dim=-2).unsqueeze(1)




class MQAttention(nn.Module):
    def __init__(self, d):
        super(MQAttention, self).__init__()

        self.linear = nn.Linear(d, d)

    def forward(self, u, m, c):
        o = attention(query=u, memory=m, context=c)

        return self.linear(o + u)


class Hop(nn.Module):
    def __init__(self, vocab_size, d):
        super(Hop, self).__init__()
        self.memory_embedder = Memory(vocab_size=vocab_size, d=d)
        self.mqattn = MQAttention(d=d)
        self.norm=LayerNorm(d)
    def forward(self, memory, query_embed):
        m, c = self.memory_embedder.forward(memory)
        query_embed

        output = self.mqattn.forward(u=query_embed, m=m, c=c)

        return self.norm(output)


class MemoryNetwork(nn.Module):
    def __init__(self, vocab_size, d, N):
        super(MemoryNetwork, self).__init__()
        self.hops = nn.ModuleList()

        self.query_embeder = Query(vocab_size=vocab_size, d=d)
        self.classifier = nn.Linear(d, vocab_size)

        for i in range(N):
            self.hops.append(Hop(vocab_size=vocab_size, d=d))

        self.N = N

    def forward(self, query, memory):
        u = self.query_embeder.forward(query)

        for h in self.hops:
            u = h.forward(memory=memory, query_embed=u)

        return self.classifier(u)
