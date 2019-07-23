import pandas as pd
import torch
import pickle
# output_model_file = "./models/swag_model_file.bin"
# output_config_file = "./models/swag_config_file.bin"
# output_vocab_file = "./models/swag_vocab_file.bin"


def load_pair(path):
    dataset=pd.read_csv(path)
    EN=[]
    ES=[]

    for i in dataset.loc[:,"EXAMPLE (ES)"]:
        ES.append(i)
    for i in dataset.loc[:,"EXAMPLE (EN)"]:
        EN.append(i)

    return ES,EN


def save(model_file_path, config_file_path, model):
    output_model_file = model_file_path
    output_config_file = config_file_path

    model_to_save = model.module if hasattr(model, 'module') else model

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

def load_pickle(path):

    with open(path, "rb") as f:
        data = pickle.load(f,)
    return data

def save_pickle(path,data):

    with open(path, "wb") as f:
        pickle.dump(data, f)
        print("save at "+ path)