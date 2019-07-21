import pandas as pd
import pickle

def load_pair(path):
    dataset=pd.read_csv(path)
    EN=[]
    ES=[]

    for i in dataset.loc[:,"EXAMPLE (ES)"]:
        ES.append(i)
    for i in dataset.loc[:,"EXAMPLE (EN)"]:
        EN.append(i)

    return ES,EN


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f,)

    return data



def save_pickle(path,data):

    with open(path, "wb") as f:
        pickle.dump(data, f)
        print("save at "+ path)