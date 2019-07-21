import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from load_csv import load_pair
from train import train
from utils import num_parameter
from preprocessing import make_dictionary,prepare_data,make_src_idx,make_trg_idx,eng_tokenize,es_tokenize

from model import EncoderGRU,Attention,DecoderGRU,Seq2Seq_a


def main():

    torch.manual_seed(777)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    parser=argparse.ArgumentParser()

    parser.add_argument("--path",type=str)
    parser.add_argument("--embedding_dim",type=int,default=300)
    parser.add_argument("--iterator",type=int,default=10)
    parser.add_argument("--lr",type=float,default=1e-5)
    parser.add_argument("--decay",type=float,default=0.01)
    parser.add_argument("--batch_size",type=int,default=100)

    args=parser.parse_args()

    trg,src=load_pair(args.path)


    src_token=eng_tokenize(src)
    trg_token=es_tokenize(trg)
    trg2idx,idx2_trg=make_dictionary(trg_token)
    src2idx,idx2src=make_dictionary(src_token)
    src_ix=make_src_idx(src_token,src2idx)
    trg_ix= make_trg_idx(trg_token,trg2idx)

    args.embedding_dim

    # model 선언부
    encoder= EncoderGRU(emb_dim=args.embedding_dim,bidirectional=True,vocab_size=len(src2idx))
    attention=Attention(emb_dim=args.embedding_dim,padding_idx=0)

    decoder=DecoderGRU(emb_dim=args.embedding_dim,attention=attention,n_class=len(trg2idx))
    model=Seq2Seq_a(encoder,decoder,device,trg2idx)

    num_parameter(model)

    #loss , optimizer 설정
    loss_func=nn.CrossEntropyLoss(ignore_index=0)
    optimizer=optim.RMSprop(model.parameters(),lr=args.lr,weight_decay=args.decay)

    #data 나누기

    train_loader,test_loader=prepare_data(src=src_ix,trg=trg_ix,test_size=0.2,batch_size=args.batch_size)
    train(model,iterator=args.iterator,optimizer=optimizer,criterion=loss_func,train_loader=train_loader,visual_path="ssibal",trg2idx=trg2idx,savepath="./seq2seq_model.pth")


if __name__ =="__main__":
    main()