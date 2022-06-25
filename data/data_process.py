import pandas as pd
import json
import pickle

def get_dataset_1(path):
    dataset_1 = pd.read_csv(path)
    dataset_1["toxic?"] = dataset_1.toxic + dataset_1.severe_toxic + dataset_1.obscene + dataset_1.threat + dataset_1.insult +dataset_1.identity_hate
    def bundle(x):
        if x==0:
            return 0
        else:
            return 1
    dataset_1["toxic?"] = dataset_1["toxic?"].apply(bundle)
    dataset_1["dataset"] = 0
    dataset_1.drop(["id","toxic","severe_toxic","obscene","threat","insult","identity_hate"],axis=1, inplace=True)
    dataset_1.rename(columns={"comment_text":"text"}, inplace=True)
    return dataset_1

dataset_1 = get_dataset_1(r"./dataset1/train.csv")
print("dataset_1 size : ", len(dataset_1))
print("dataset_1 toxic count : ", dataset_1["toxic?"].sum())

def get_dataset_2(path):
    dataset_2 = pd.read_pickle(path)
    def unify(x):
        if x==2:
            return 0
        else:
            return 1
    dataset_2["toxic?"] = dataset_2["class"].apply(unify)
    dataset_2["dataset"] = 1
    dataset_2.rename(columns={"tweet":"text"}, inplace=True)
    dataset_2.drop(["count","hate_speech","offensive_language","neither","class"],axis=1, inplace=True)
    return dataset_2

dataset_2 = get_dataset_2(r"2-hate_speech_dataset_davidson.p")
print("dataset_2 size : ", len(dataset_2))
print("dataset_2 toxic count : ", dataset_2["toxic?"].sum())


def get_dataset_3(path):
    dataset_3 = pd.read_excel(path)
    def unify(x):
        if x=="N":
            return 0
        else:
            return 1
    dataset_3["toxic?"] = dataset_3["Code"].apply(unify)
    dataset_3["dataset"] = 2
    dataset_3.rename(columns={"Tweet":"text"}, inplace=True)
    dataset_3.drop(["ID","Code"], axis=1, inplace=True)
    return dataset_3
try:
    dataset_3 = get_dataset_3(r"./dataset3/onlineHarassmentDataset.xlsx")
    print("dataset_3 size : ", len(dataset_3))
    print("dataset_3 toxic count : ", dataset_3["toxic?"].sum())
except:
    print("Please request dataset of Golbeck-et-al")
    print("You can find required information at https://doi.org/10.1145/3091478.3091509")
    exit(1)

def get_dataset_4(path):
    dataset_4 = pd.read_csv(path)
    dataset_4.rename(columns={"Insult":"toxic?","Comment":"text"}, inplace=True)
    dataset_4.drop("Date", axis=1, inplace=True)
    dataset_4["dataset"] = 3
    return dataset_4

dataset_4 = get_dataset_4(r"./dataset4/train.csv")
print("dataset_4 size : ", len(dataset_4))
print("dataset_4 toxic count : ", dataset_4["toxic?"].sum())

def merge_datasets():
    df=dataset_1.append(dataset_2)
    df=df.append(dataset_3)
    df=df.append(dataset_4)
    df.reset_index(inplace=True,drop=True)
    return df
sentence_level_df = merge_datasets()
print("sentence_level_df size : ", len(sentence_level_df))
print("sentence_level_df toxic count : ", sentence_level_df["toxic?"].sum())


sentence_level_df.to_csv(r"sentence_level_df.txt",index=False)