import torch.optim as optim
import torch
import torch.nn as nn

import argparse


from load_csv import load_pair
from preprocessing import tokenize,make_dictionary,src_convert,trg_convert,prepare_data
from model import EncoderDecoder
from train import train

torch.manual_seed(777)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


import pandas as pd

parser=argparse.ArgumentParser()


parser.add_argument("-path", type=str, help="directory for reading file",default="C://Users/Jimmy Hong/PycharmProjects/NMT/Data/translation_pair.csv")

parser.add_argument('-head', type=int, help="How many number of Layers?", default=8)
parser.add_argument('-d_model', type=int, help="magnitude of embedding dimension?", default=512)
parser.add_argument('-d_ff', type=int, help="How many will iterate?", default=2048)

parser.add_argument('-N', type=int, help="how many stack>", default=6)

parser.add_argument('-iterator', type=int, help="How many will iterate?", default=10)

parser.add_argument('-test_size', type=float, help="test_size", default=.2)
parser.add_argument('-lr', type=float, help="learning_rate", default=1e-3)
parser.add_argument('-weight_decay', type=float, help="weight_decay", default=1e-5)
parser.add_argument('-batch_size', type=int, help="batch size", default=12)


def main():
    args=parser.parse_args()
    trg,src=load_pair(args.path)


    src_ix=tokenize(src)

    tgt_ix=tokenize(trg)

    eng2idx,idx2eng=make_dictionary(src_ix)
    tgt2idx,idx2es=make_dictionary(tgt_ix)


    src_sentences=src_convert(src_ix,eng2idx)
    tgt_sentences=trg_convert(tgt_ix,tgt2idx)

    train_loader,test_loader=prepare_data(src=src_sentences,trg=tgt_sentences,test_size=args.test_size,batch_size=12)
    transformer=EncoderDecoder(d_model=args.d_model,h=args.head,src_num=len(eng2idx),tgt_num=len(tgt2idx),N=args.N,device=device,d_ff=args.d_ff)


    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(transformer.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    train(model=transformer,iterator=args.iterator,optimizer=optimizer,criterion=criterion,
          train_loader=train_loader,teacher_force=True,test_loader=test_loader,device=device)


if __name__=="__main__":
    main()