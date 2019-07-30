import torch
import torch.nn as nn
from model import Maxout

device= "cpu"




class HighwayMaxoutNetwork(nn.Module):
    def __init__(self, d_model, pooling_size=2):
        super(HighwayMaxoutNetwork, self).__init__()
        self.mlp = nn.Linear(d_model, d_model)

        self.d_model = d_model

        self.lstm = nn.LSTMCell(input_size=d_model * 4, hidden_size=d_model)

        self.Wd_start = nn.Linear(d_model * 5, d_model)
        self.Wd_end = nn.Linear(d_model * 5, d_model)

        self.max_start0 = Maxout(dim_in=d_model * 3, dim_out=d_model, pooling_size=pooling_size)
        self.max_start1 = Maxout(dim_in=d_model, dim_out=d_model, pooling_size=pooling_size)
        self.max_start2 = Maxout(dim_in=d_model * 2, dim_out=1, pooling_size=pooling_size)

        self.max_end0 = Maxout(dim_in=d_model * 3, dim_out=d_model, pooling_size=pooling_size)
        self.max_end1 = Maxout(dim_in=d_model, dim_out=d_model, pooling_size=pooling_size)
        self.max_end2 = Maxout(dim_in=d_model * 2, dim_out=1, pooling_size=pooling_size)

    def forward(self, ut, start_state, end_state, hidden, cell):

        con_lens = ut.size(1)
        n_batches = ut.size(0)
        dim=ut.size(2)

        alpha, beta = torch.tensor([], requires_grad=True).to(device), torch.tensor([], requires_grad=True).to(device)

        h_s_e = torch.cat([hidden, start_state, end_state], dim=-1)

        r = torch.tanh(self.Wd_start.forward(h_s_e))

        start, end = torch.LongTensor([0] * n_batches), torch.LongTensor([0] * n_batches).to(device)

        for j in range(con_lens):
            j_th_states = ut[:, j, :]
            e_r = torch.cat([j_th_states, r], dim=-1)
            m1 = self.max_start0.forward(e_r)
            m2 = self.max_start1.forward(m1)
            hmn = self.max_start2.forward(torch.cat([m1, m2], dim=-1))
            alpha = torch.cat([alpha, hmn], dim=-1)

        start = torch.argmax(alpha, dim=-1)
        s1 = start.view(-1, 1, 1).expand(n_batches, 1, dim)
        start_state = ut.gather(dim=1, index=s1).view(n_batches, -1)
        h_s_e = torch.cat([hidden, start_state, end_state], dim=-1)

        for j in range(con_lens):
            j_th_states = ut[:, j, :]
            e_r = torch.cat([j_th_states, r], dim=-1)

            m1 = self.max_end0.forward(e_r)
            m2 = self.max_end1.forward(m1)
            hmn = self.max_end2.forward(torch.cat([m1, m2], dim=-1))

            beta = torch.cat([beta, hmn], dim=-1)

        end = torch.argmax(beta, dim=-1)
        e1 = end.view(-1, 1, 1).expand(n_batches, 1, dim)
        end_state = ut.gather(dim=1, index=e1).view(n_batches, -1)

        hidden, cell = self.lstm(torch.cat([start_state, end_state], dim=-1), (hidden, cell))
        hidden = self.mlp.forward(hidden)

        return alpha, beta,hidden,cell,start_state,end_state,start,end


class Decoder(nn.Module):
    def __init__(self, d_model, iters, pooling_size=2):
        super(Decoder, self).__init__()

        self.iters = iters

        self.HMN = HighwayMaxoutNetwork(d_model=d_model, pooling_size=pooling_size)

    def forward(self, ut):
        con_lens = ut.size(1)
        n_batches = ut.size(0)
        dim = ut.size(2)

        hx, cx = torch.zeros([n_batches, dim // 2]).to(device), torch.zeros([n_batches, dim // 2]).to(device)

        start, end = torch.LongTensor([0] * n_batches).to(device), torch.LongTensor([0] * n_batches).to(device)

        s1 = start.view(-1, 1, 1).expand(n_batches, 1, dim)
        e1 = start.view(-1, 1, 1).expand(n_batches, 1, dim)

        start_state = ut.gather(dim=1, index=s1).view(n_batches, -1)
        end_state = ut.gather(dim=1, index=e1).view(n_batches, -1)

        entropies = []

        for i in range(self.iters):

            alpha, beta,hidden,cell,start_state,end_state,start,end=self.HMN.forward(ut,start_state=start_state,end_state=end_state,hidden=hx,cell=cx)

            entropies.append([alpha, beta])

        return start, end, entropies


class DynamicCN(nn.Module):
    def __init__(self, d_model, vocab_size, iters):
        super(DynamicCN, self).__init__()
        self.coattn = Co_Attention(d_model=d_model)
        self.encoder = Encoder(d_model=d_model, vocab_size=vocab_size)
        self.decoder = Decoder(d_model=d_model, iters=iters, pooling_size=2)

    def forward(self, question, context):
        q, c = self.forward(question=question, context=context)
        ut = self.coattn.forward(query=q, context=c, query_batch=question, context_batch=context)

        s, e, entropies = self.decoder.forward(ut)

        return s, e, entropies
