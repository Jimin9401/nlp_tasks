
from utils import makeData
from torch.utils.data import DataLoader,Dataset
from utils import pad_sequence


from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split
#from utils import makeData
#from torch.utils.data import DataLoader,Dataset
#from utils import pad_sequence



def tokenize(sentence):

    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    token_sentence=[]
    for i in sentence:
        token_sentence.append(tokenizer.tokenize(i))

    return token_sentence

def make_dictionary(tokens):
    flattens=[]
    for i in tokens:
        for k in i:
            flattens.append(k)

    word2idx={"[PAD]":0,"[UNK]":1,"[EOS]":2,"[START]":3}

    idx2word= { 0:"[PAD]", 1:"[UNK]",2:"[EOS]",3:"[START]"}

    ix=4
    for i in flattens:
        if i not in word2idx:
            word2idx[i]=ix
            idx2word[ix]=i
            ix+=1

    return word2idx,idx2word


def src_convert(seq_token,word2idx):

    seq_ix=[]

    for i in seq_token:
        tmp_ix=[]
        for k in i:
            tmp_ix.append(word2idx[k])
        seq_ix.append(tmp_ix)

    return seq_ix

def trg_convert(seq_token,word2idx):

    seq_ix=[]

    for i in seq_token:
        tmp_ix=[3,]

        for k in i:
            tmp_ix.append(word2idx[k])
        tmp_ix.append(2)
        seq_ix.append(tmp_ix)

    return seq_ix



def prepare_data(src,trg,test_size,batch_size):
    X_train,X_test,y_train,y_test=train_test_split(src,trg,test_size=test_size,random_state=777)

    train_data=makeData(X_train,y_train)
    test_data=makeData(X_test,y_test)

    train_loader=DataLoader(dataset=train_data,collate_fn=pad_sequence,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_data,collate_fn=pad_sequence,batch_size=len(test_data))

    return train_loader,test_loader