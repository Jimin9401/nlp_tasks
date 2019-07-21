from torch.utils.data import Dataset, DataLoader
import torch


class makeData(Dataset):
    def __init__(self, X_data, y_data,vocab):
        self.X_data = X_data
        self.y_data = y_data
        self.vocab=vocab
    def __getitem__(self, index):
        return self.X_data[index], [self.vocab["[START]"]]+self.y_data[index]+[self.vocab["[EOS]"]]

    def __len__(self):
        return len(self.X_data)

def pad_sequence(batch):
    X_batch, y_batch = zip(*batch)
    batch_size=len(X_batch)
    X_max_seq_length = max([len(x) for x in X_batch])
    X_res = []
    for seq in X_batch:
        if len(seq) < X_max_seq_length:
            pad_seq = torch.LongTensor(seq + [0] * (X_max_seq_length - len(seq)))
            X_res.append(pad_seq)
        else:
            X_res.append(torch.LongTensor(seq))

    y_max_seq_length = max([len(x) for x in y_batch])
    y_res = []
    for seq in y_batch:
        if len(seq) < y_max_seq_length:
            pad_seq = torch.LongTensor(seq + [0] * (y_max_seq_length - len(seq)))
            y_res.append(pad_seq)
        else:
            y_res.append(torch.LongTensor(seq))

    return torch.cat(X_res).reshape(batch_size, X_max_seq_length), torch.cat(y_res).reshape(batch_size,y_max_seq_length)


def num_parameter(model):
    sum=0
    for p in model.parameters():
        if p.requires_grad:
            sum+=p.numel()
    return sum