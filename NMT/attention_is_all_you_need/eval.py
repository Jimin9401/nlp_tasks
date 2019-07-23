
import copy
import torch
import numpy as np


def exact_match(seq_generated, seq_true):
    return torch.sum(seq_generated[seq_generated != 0] == seq_true[seq_generated != 0]).item() / (seq_true[
        seq_generated != 0].size(0)+1e-5)

def greedy_decode(x, key, value, decoder, device, src, trg, n_class):
    tgt = copy.deepcopy(trg)
    batch_size, tgt_lens = tgt.size()
    score = torch.zeros([batch_size, tgt_lens, n_class]).to(device)

    for t in range(0, tgt_lens - 1):
        tgt_masked = torch.tril(tgt.unsqueeze(1), diagonal=t).squeeze(1)
        y = decoder.forward(x=tgt_masked, key=key, value=value, time=t, src=src, tgt=tgt_masked)

        score[:, t + 1, :] = y

        tgt[:, t + 1] = y.data.topk(1)[-1].squeeze(-1)

    return score, tgt


def evaluate(test_loader, criterion, model, device):
    average_loss = []
    match_score_list = []
    model = model.eval()

    for k, (src_batch, trg_batch) in enumerate(test_loader):
        src_tensor = torch.LongTensor(src_batch).to(device)
        trg_tensor = torch.LongTensor(trg_batch).to(device)
        with torch.no_grad():
            scores, seq_generated = model.forward(src=src_tensor, trg=trg_tensor, train=False)
            match = exact_match(seq_generated=seq_generated[:, 1:], seq_true=trg_tensor[:, 1:])
            match_score_list.append(match)
            scores = scores[:, 1:].contiguous().view(-1, scores.shape[-1])
            trg_tensor = trg_tensor[:, 1:].contiguous().view(-1)

            loss = criterion(scores, trg_tensor)
            average_loss.append(loss.item())

    print("----------------------------------------")
    print("Eval Average Loss: {:.4f} +- {:.4f}".format(np.mean(average_loss), np.std(average_loss)))
    print("Eval Exact match score: {:.4f} +- {:.4f}".format(np.mean(match_score_list), np.std(match_score_list)))
    print("----------------------------------------")




def test(test_loader, model, device,s_index):
    model = model.eval()
    total_generated = []
    for k, (src_batch, trg_batch) in enumerate(test_loader):
        src_tensor = torch.LongTensor(src_batch).to(device)
        trg_tensor = torch.LongTensor(torch.LongTensor([[s_index] * 20] * src_tensor.size(0))).to(device)

        with torch.no_grad():
            _, seq_generated = model.forward(src=src_tensor, tgt=trg_tensor, train=False)
            total_generated.append(seq_generated)

    return total_generated