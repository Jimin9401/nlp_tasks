import torch

import torch.nn as nn
import torch.optim as optim
from IOfile import load_pair
from train import train
from preprocessing import make_dictionary,prepare_data,eng_tokenize,es_tokenize,convert
from model import ConvS2S

import argparse

torch.manual_seed(777)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

## arguments in console
parser.add_argument("path", type=str, help="directory for reading file")
parser.add_argument('num_of_layer', type=int, help="How many number of Layers?", default=15)
parser.add_argument('embedding_dim', type=int, help="magnitude of embedding dimension?", default=512)
parser.add_argument('iterator', type=int, help="How many will iterate?", default=10)
parser.add_argument('test_size', type=float, help="test_size", default=.2)
parser.add_argument('weight_decay', type=float, help="weight_decay", default=1e-3)
parser.add_argument('batch_size', type=int, help="batch size", default=64)
args = parser.parse_args()


def main():

    ######### args parameters

    trg,src=load_pair(args.path)
    src_token=eng_tokenize(src)
    trg_token=es_tokenize(trg)

    ###############################################
    trg2idx,idx2trg=make_dictionary(trg_token)

    src2idx,idx2src=make_dictionary(src_token)

    src_convert=convert(word2idx=src2idx,idx2word=idx2src)

    trg_convert=convert(word2idx=trg2idx,idx2word=idx2trg)

    src_ix=src_convert.from_seq2idx(src_token)

    trg_ix= trg_convert.from_seq2idx(trg_token)

    train_loader,test_loader=prepare_data(src=src_ix,trg=trg_ix,test_size=args.test_size,batch_size=args.batch_size,y_vocab=trg2idx)

    #loss , optimizer 설정
    loss_func=nn.CrossEntropyLoss(ignore_index=0)
    model=ConvS2S(src_size=len(src2idx),tgt_size=len(trg2idx),N=args.num_of_layer,d_vector=512,k_size=3,device=device)
    optimizer=optim.Adam(model.parameters(),weight_decay=args.weight_decay)

    train(model=model,iterator=args.iterator,optimizer=optimizer,criterion=loss_func,train_loader=train_loader,test_loader=test_loader)

    ############

if __name__=="__main__":
    main()