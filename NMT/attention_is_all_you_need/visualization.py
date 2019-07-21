from tensorboardX import SummaryWriter
import torch
import numpy as np


def visualize(loss,iteration,path,model,src,trg,trg2idx):
    writer= SummaryWriter(path)
    writer.add_scalar("Train/Loss",loss,iteration)

    # with SummaryWriter(comment="Seq2Seq") as w:
    #     w.add_graph(model,(torch.Tensor(1,len(trg2idx)),src,trg))
