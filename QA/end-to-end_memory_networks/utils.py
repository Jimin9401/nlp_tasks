from torch.utils.data import Dataset, DataLoader
import torch
from torchtext.datasets import BABI20

def dataloader(batch_size, memory_size, task, joint, tenK):
    train_iter, valid_iter, test_iter = BABI20.iters(
        batch_size=batch_size, memory_size=memory_size, task=task, joint=joint, tenK=tenK, device=torch.device("cpu"))
    return train_iter, valid_iter, test_iter, train_iter.dataset.fields['query'].vocab


class makeBatch:
    def __init__(self,q,c,s_idx,e_idx,padding_idx=0):
        self.q=q
        self.c=c

        self.s_idx=s_idx
        self.e_idx=e_idx

    def __getitem__(self, index):
        return self.q[index], self.c[index],self.s_idx[index],self.e_idx[index]

    def __len__(self):
        return len(self.q)


def pad_sequence(batch):

    Q_batch, C_batch,s_ix,e_ix = zip(*batch)

    batch_size=len(Q_batch)


    Q_max_seq_length = max([len(x) for x in Q_batch])
    Q_res = []

    for seq in Q_batch:
        if len(seq) < Q_max_seq_length:
            pad_seq = torch.LongTensor(seq + [0] * (Q_max_seq_length - len(seq)))
            Q_res.append(pad_seq)
        else:
            Q_res.append(torch.LongTensor(seq))
    ##
    C_max_seq_length = max([len(x) for x in C_batch])
    C_res = []

    for seq in C_batch:
        if len(seq) < C_max_seq_length:
            pad_seq = torch.LongTensor(seq + [0] * (C_max_seq_length - len(seq)))
            C_res.append(pad_seq)
        else:
            C_res.append(torch.LongTensor(seq))

    return torch.cat(Q_res).reshape(batch_size, Q_max_seq_length), torch.cat(C_res).reshape(batch_size,C_max_seq_length),\
          torch.LongTensor(s_ix).reshape(batch_size),torch.LongTensor(e_ix).reshape(batch_size)



def num_parameter(model):
    sum=0
    for p in model.parameters():
        if p.requires_grad:
            sum+=p.numel()
    return sum