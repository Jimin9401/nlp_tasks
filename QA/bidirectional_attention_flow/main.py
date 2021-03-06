from IOfile import load_pickle,save_pickle
import torch
import torch.nn as nn
import torch.nn.functional as  F
import torch.optim as optim
import argparse

#####
from utils import makeBatch,pad_sequence
from preprocessing import make_dictionary,convert2idx,find_sub_list,tokenize,preprocess_data,pad_sequence
from model import WordEmbedder,Encoder,AttentionFlow,attention,ModelingLayer,OutputLayer,BIDAF
from torch.utils.data import Dataset,DataLoader
from train import train


torch.manual_seed(777)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--path",help="address of file",type=str)

    parser.add_argument("--batch_size",help="batch_size",type=int,default=12)
    parser.add_argument("--embedding_size",help="dimension of vectors",default=300,type=int)
    parser.add_argument("--lr",type=float, help="learning rate",default=1e-5)
    parser.add_argument("--decay",help="L2 loss",type=float,default=1e-2)
    parser.add_argument("--iterator",type=int, help="number of iteration",default=10)

    args=parser.parse_args()

    data=load_pickle(args.path)

    context=data["context"]
    question=data["question"]
    answer=data["answer"]

    cxt=[]
    query=[]
    ans=[]

    for c,q,a in zip(context,question,answer):
        cxt.append(c.lower())
        query.append(q.lower())
        ans.append(a.lower())

    cxt=tokenize(cxt)
    query=tokenize(query)
    ans=tokenize(ans)

    word2idx,idx2word=make_dictionary(cxt,query,ans)

    query_ix=convert2idx(query,word2idx)
    context_ix=convert2idx(cxt,word2idx)
    answer_ix=convert2idx(ans,word2idx)

    ##preprocess data
    q_data,c_data,a_data,start_index,end_index=preprocess_data(query_ix,context_ix,answer_ix)

    train_data= makeBatch(q_data,c_data, start_index, end_index)

    train_loader=DataLoader(train_data,collate_fn=pad_sequence,batch_size=args.batch_size)
    ################################################################################################

    ## train
    model=BIDAF(embedder=WordEmbedder(args.embedding_size,len(word2idx)),encoder=Encoder(args.embedding_size,args.embedding_size),
                attention_flow=AttentionFlow(),modeling_layer=ModelingLayer(d_vector=args.embedding_size,bidirectional=True),
                output_layer=OutputLayer(d_vector=args.embedding_size)).to(device)


    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.decay)

    train(model,args.iterator,optimizer,criterion,train_loader)

if __name__=="__main__":
    main()