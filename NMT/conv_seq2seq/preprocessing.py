from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split
from utils import makeData
from torch.utils.data import DataLoader,Dataset
from utils import pad_sequence
from nltk.tokenize import WhitespaceTokenizer


def eng_tokenize(sentence):

    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    token_sentence=[]
    for i in sentence:
        token_sentence.append(tokenizer.tokenize(i))

    return token_sentence

def es_tokenize(sentence):

    tokenizer=WhitespaceTokenizer()
    token_sentence=[]

    for i in sentence:
        token_sentence.append(tokenizer.tokenize(i))

    return token_sentence


class convert():
    def __init__(self,word2idx,idx2word):
        super(convert).__init__()
        self.word2idx=word2idx
        self.idx2word=idx2word

    def from_seq2idx(self,inputs):
        seq_ix = []
        for i in inputs:
            tmp_ix = []
            for k in i:
                tmp_ix.append(self.word2idx[k])
            seq_ix.append(tmp_ix)
        return seq_ix

    def from_idx2seq(self,inputs):
        seq_token = []
        for i in inputs:
            tmp_token = []
            for k in i:
                tmp_token.append(self.idx2word[k])
            seq_token.append(tmp_token)
        return seq_token

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

def prepare_data(src,trg,test_size,batch_size,y_vocab):
    X_train,X_test,y_train,y_test=train_test_split(src,trg,test_size=test_size,random_state=777)

    train_data=makeData(X_train,y_train,y_vocab)
    test_data=makeData(X_test,y_test,y_vocab)

    train_loader=DataLoader(dataset=train_data,collate_fn=pad_sequence,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_data,collate_fn=pad_sequence,batch_size=batch_size)

    return train_loader,test_loader