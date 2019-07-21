import pandas as pd

def load_pair(path):
    dataset=pd.read_csv(path)
    EN=[]
    ES=[]

    for i in dataset.loc[:,"EXAMPLE (ES)"]:
        ES.append(i)
    for i in dataset.loc[:,"EXAMPLE (EN)"]:
        EN.append(i)

    return ES,EN