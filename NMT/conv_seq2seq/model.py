import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from eval import greedy_decode

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def clones(N, layer):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


def attention(states, tgt, mask=None, dropout=None):
    scores = torch.matmul(states, tgt.transpose(-2, -1))

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

    p_attn = F.softmax(scores, dim=-2)
    return torch.matmul(states.transpose(1, 2), p_attn).transpose(1, 2)


def position_encoding_init(n_position, emb_dim):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor).to(device)


class Convlayer(nn.Module):
    def __init__(self, k_size, d_vector, decode=False):
        super(Convlayer, self).__init__()
        self.d_vector = d_vector
        if decode:
            self.convlayer = nn.Conv1d(in_channels=d_vector, out_channels=2 * d_vector, kernel_size=k_size,
                                       padding=0, stride=1)

        else:
            self.convlayer = nn.Conv1d(in_channels=d_vector, out_channels=2 * d_vector, kernel_size=k_size,
                                       padding=int((k_size - 1) / 2), stride=1)

        self.norm = nn.BatchNorm1d(d_vector)

    def forward(self, x):
        a, b = self.convlayer.forward(x.transpose(1, 2)).chunk(2, dim=1)

        return F.relu(self.norm((a * torch.sigmoid(b)))).transpose(1, 2)


class stackedConv(nn.Module):
    def __init__(self, N, d_vector, k_size, decode=False):
        super(stackedConv, self).__init__()
        self.convlayers = nn.ModuleList()
        self.decode = decode
        for _ in range(N):
            self.convlayers.append(Convlayer(k_size=k_size, d_vector=d_vector, decode=decode))
        self.k_size = k_size

    def forward(self, x):
        res = None
        padding = torch.zeros(x.shape[0], self.k_size - 1, x.size(2)).to(device)
        if self.decode:
            for c in self.convlayers:
                x = torch.cat([padding, x], dim=1)
                x = c.forward(x) * np.sqrt(0.5)
                if res is not None:
                    x += res
                res = x
        else:
            for c in self.convlayers:
                x = c.forward(x) * np.sqrt(0.5)
                if res is not None:
                    x += res
                res = x
        return x


class ConvEncoder(nn.Module):
    def __init__(self, k_size, d_vector, vocab_size, N, dropout=0.1):
        super(ConvEncoder, self).__init__()

        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_vector, padding_idx=0)
        self.d_vector = d_vector
        self.convlayers = stackedConv(N=N, d_vector=d_vector, k_size=k_size, decode=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.dropout(self.embedder.forward(inputs))
        seq_lens = x.size(1)
        dim = x.size(2)

        x += position_encoding_init(n_position=seq_lens, emb_dim=dim)

        x = self.convlayers.forward(x)

        return x


class Attention(nn.Module):
    def __init__(self, padding_idx=0):
        super(Attention, self).__init__()
        self.padding_idx = padding_idx

    def forward(self, state, x, src):
        mask = (src == self.padding_idx)
        w = attention(state, x, mask=mask)

        w[mask] = float("-inf")
        attn_score = F.softmax(w, dim=1)

        return torch.matmul(state.transpose(1, 2), attn_score).transpose(1, 2)


class ConvDecoder(nn.Module):

    def __init__(self, k_size, d_vector, vocab_size, N, dropout=0.1):
        super(ConvDecoder, self).__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_vector, padding_idx=0)
        self.d_vector = d_vector
        self.convlayers = stackedConv(N=N, d_vector=d_vector, k_size=k_size, decode=True)
        self.attn = attention
        self.classifier = nn.Linear(2 * d_vector, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, states, src, t, train):

        tgt = self.dropout(self.embedder.forward(tgt))

        seq_lens = tgt.size(1)
        dim = tgt.size(2)

        tgt += position_encoding_init(n_position=seq_lens, emb_dim=dim)

        tgt = self.convlayers.forward(tgt)

        seq_lens = tgt.size(1)

        if train:
            tgt = tgt
            states = F.adaptive_avg_pool1d(states, tgt.size(2))
            w = self.attn(states=states, tgt=tgt, mask=src)
            y = self.classifier(torch.cat([tgt, w], dim=2))

            return y

        else:
            tgt = tgt[:, t - 1, :].unsqueeze(1)

            states = F.adaptive_avg_pool1d(states, tgt.size(2))

            w = self.attn(states=states, tgt=tgt, mask=src)

            y = self.classifier(torch.cat([tgt, w], dim=2)).squeeze(1)

            return F.softmax(y, dim=1)


class ConvS2S(nn.Module):
    def __init__(self, src_size, tgt_size, N, d_vector, k_size, device):
        super(ConvS2S, self).__init__()
        self.encoder = ConvEncoder(k_size=k_size, d_vector=d_vector, vocab_size=src_size, N=N)
        self.decoder = ConvDecoder(k_size=k_size, d_vector=d_vector, vocab_size=tgt_size, N=N)
        self.tgt_size = tgt_size
        self.device = device

    def forward(self, src, tgt, train=True):
        states = self.encoder.forward(src)

        n_batch = tgt.size(0)
        tgt_lens = tgt.size(1)

        if train:
            score = torch.zeros([n_batch, tgt_lens, self.tgt_size]).to(self.device)
            y = self.decoder.forward(tgt=tgt, states=states, src=src, t=None, train=True)
            score[:, 1:, :] = y[:, :tgt_lens - 1, :]
            return score
        else:
            score,seq_generated=greedy_decode(states=states,trg=tgt,decoder=self.decoder,
                                              device=self.device,src=src,n_class=self.tgt_size)


            return score,seq_generated
