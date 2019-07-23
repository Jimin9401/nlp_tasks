import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import numpy as np
import math
from eval import greedy_decode


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def attention(query, key, value, mask):
    w = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    mask = mask.unsqueeze(1)
    w = w.masked_fill(mask == 0, -1e9)
    attn_score = F.softmax(w, dim=-1)

    #    print(attn_score)

    return torch.matmul(attn_score, value), attn_score


def position_encoding_init(n_position, emb_dim):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim

    return torch.from_numpy(position_enc).type(torch.FloatTensor).to(device)


def clones(module, N):
    layers = nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    return layers


class Embedder(nn.Module):

    def __init__(self, d_model, vocab_size, padding_idx=0):
        super(Embedder, self).__init__()
        self.embedding = nn.Embedding(padding_idx=padding_idx, num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        # x = [ batch , seq_lens]
        x_embed = self.embedding(x)
        seq_lens = x_embed.size(1)
        embed_dim = x_embed.size(2)

        # x= [batch,seq_lens, d_model]
        x_embed += position_encoding_init(n_position=seq_lens, emb_dim=embed_dim)

        return x_embed


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.d_model = d_model
        self.norm1 = nn.Parameter(torch.ones(d_model))
        self.norm2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.norm1 * (x - mean) / (std + self.eps) + self.norm2


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super(MultiHeadAttention, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.d_model = d_model
        self.d_k = d_model // h

        self.h = h
        self.attention = attention

    def forward(self, query, key, value, batch):
        n_batches = query.size(0)

        query, key, value = [l.forward(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        z, attn = attention(query=query, key=key, value=value, mask=batch)

        z = z.transpose(1, 2).contiguous().view(n_batches, -1, self.d_model)

        return self.linears[-1].forward(z), attn


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=2048, p=0):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=p)

    def forward(self, z):
        return self.linear2.forward(self.dropout(F.relu(self.linear1.forward(z))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super(EncoderBlock, self).__init__()
        self.h = h
        self.multihead = MultiHeadAttention(h, d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        z, attn = self.multihead(x, x, x, mask.unsqueeze(1))

        z = self.ffn.forward(z)

        return self.norm(x + z)


class Encoder(nn.Module):
    def __init__(self, d_model, h, vocab_size, d_ff, N=6):
        super(Encoder, self).__init__()
        self.blocks = clones(EncoderBlock(d_model=d_model, h=h, d_ff=d_ff), N)
        self.embedder = Embedder(d_model=d_model, vocab_size=vocab_size, padding_idx=0)

    def forward(self, x):
        mask = x
        x = self.embedder.forward(x)

        #       print("====encoder attention====")
        for b in self.blocks:
            x = b.forward(x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, eps=1e-6):
        super(DecoderBlock, self).__init__()
        self.h = h
        self.multiheads = clones(MultiHeadAttention(h, d_model), 2)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layernorm = LayerNorm(d_model=d_model)

    def forward(self, x, key, value, src, tgt):
        #      print("=====decoder attention====")

        z, self_attn = self.multiheads[0](x, x, x, tgt.unsqueeze(1))

        #     print("====encoder-decoder attention=====")

        z, key_attn = self.multiheads[1](x, key, value, src.unsqueeze(1))

        z = self.ffn.forward(z)

        return self.layernorm(x + z)




class Decoder(nn.Module):
    def __init__(self, d_model, h, vocab_size, d_ff, N=6):
        super(Decoder, self).__init__()
        self.blocks = clones(DecoderBlock(d_model=d_model, h=h, d_ff=d_ff), N)
        self.embedder = Embedder(d_model=d_model, vocab_size=vocab_size, padding_idx=0)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, key, value, time, src, tgt):
        x = self.embedder.forward(x)

        for b in self.blocks:
            x = b.forward(x, key, value, src, tgt)

        return F.log_softmax(self.classifier(x), dim=-1)[:, time, :]


class EncoderDecoder(nn.Module):
    def __init__(self, d_model, src_num, tgt_num, device, d_ff, N=6, h=8):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(N=N, d_model=d_model, h=h, vocab_size=src_num, d_ff=d_ff)
        self.decoder = Decoder(N=N, d_model=d_model, h=h, vocab_size=tgt_num, d_ff=d_ff)
        self.n_class = tgt_num
        self.device = device

    def forward(self, src, trg, train=True):
        src_z = self.encoder.forward(x=src)
        n_batches = src.size(0)
        trg_lens = trg.size(1)

        predict = torch.zeros(n_batches, trg_lens, self.n_class)

        if train:

            for i in range(0, trg_lens - 1):
                trg_masked = torch.tril(trg.unsqueeze(1), diagonal=i).squeeze(1)

                predict[:, i + 1, :] = self.decoder.forward(x=trg_masked, key=src_z, value=src_z, time=i, src=src,
                                                            tgt=trg_masked)
            #             print(predict)
            return predict

        else:
            score, seq_generated = greedy_decode(x=trg, key=src_z, value=src_z, decoder=self.decoder,
                                                 device=self.device,
                                                 src=src, trg=trg, n_class=self.n_class)

            return score, seq_generated
