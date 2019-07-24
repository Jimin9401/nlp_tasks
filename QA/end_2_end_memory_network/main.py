from utils import dataloader
import torch
import torch.nn as nn
import argparse

import torch.optim as optim
from model import MemoryNetwork
from train import train

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-batch_size", type=int, default=30)
    parser.add_argument("-N", type=int, default=3)
    parser.add_argument("-embedding_dim", type=int, default=512)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-weight_decay", type=float, default=1e-5)
    parser.add_argument("-iterator", type=int, default=30)
    args = parser.parse_args()

    train_batch,val,test,vocab=dataloader(batch_size=args.batch_size,memory_size=5,task=1,joint=False,tenK=False)


    model=MemoryNetwork(vocab_size=len(vocab),d=args.embedding_dim,N=args.N)


    criterion=nn.CrossEntropyLoss()

    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)



    train(model=model,iterator=args.iterator,optimizer=optimizer,criterion=criterion,train_loader=train_batch,test_loader=None)



if __name__=="__main__":
    main()
