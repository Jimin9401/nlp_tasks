from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split
from utils import makeData
from torch.utils.data import DataLoader,Dataset
from utils import pad_sequence


batch_size=100

def tokenize(sentence):

    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    token_sentence=[]
    for i in sentence:
        token_sentence.append(tokenizer.tokenize(i))

    return token_sentence

def make_dictionary(question,context,answer):

    tokens=[]
    tokens.append(question)
    tokens.append(context)
    tokens.append(answer)

    flattens=[]
    for i in tokens:
        for k in i:
            for j in k:
                flattens.append(j)

    word2idx={"[PAD]":0,"[UNK]":1,"[EOS]":2,"[START]":3}

    idx2word= { 0:"[PAD]", 1:"[UNK]",2:"[EOS]",3:"[START]"}

    ix=4
    for i in flattens:
        if i not in word2idx:
            word2idx[i]=ix
            idx2word[ix]=i
            ix+=1

    return word2idx,idx2word


def convert2idx(seq_token,word2idx):

    seq_ix=[]

    tmp_ix=[]
    for i in seq_token:
        for k in i:
            tmp_ix.append(word2idx[k])
        seq_ix.append(tmp_ix)
        tmp_ix=[]

    return seq_ix


def prepare_data(src,trg,test_size):
    X_train,X_test,y_train,y_test=train_test_split(src,trg,test_size=test_size,random_state=777)

    train_data=makeData(X_train,y_train)
    test_data=makeData(X_test,y_test)

    train_loader=DataLoader(dataset=train_data,collate_fn=pad_sequence,batch_size=batch_size)
    test_loader=DataLoader(dataset=test_data,collate_fn=pad_sequence,batch_size=1)

    return train_loader,test_loader

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results


def preprocess_data(query_ix,context_ix,answer_ix):
    result = []
    ix = 0
    remove_list = []
    start_index = []
    end_index = []
    q_data=[]
    c_data=[]
    a_data=[]
    for q,c, a in zip(query_ix,context_ix, answer_ix):
        if (find_sub_list(a, c) != []):
            q_data.append(q)
            c_data.append(c)
            a_data.append(a)
            result.append(find_sub_list(a,c)[0])
            remove_list.append(ix)

    for st_ix, e_ix in result:
        start_index.append(st_ix)
        end_index.append(e_ix)

    return q_data, c_data, a_data, start_index, end_index
