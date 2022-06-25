import numpy as np
import pandas as pd
import os, pickle
import torch
from transformers import BertTokenizer,BertModel, AdamW
from torch import nn
from torch.utils.data import Dataset, DataLoader

print("Sentence Classifier...")
print("This is the first step of Sequential Adaption approach\n")

df = pd.read_csv("./data/sentence_level_df.txt")

#Project trial
"""
TRIAL -1
df = df[df["dataset"] != 0]
TRIAL -2
df = df[df["dataset"] != 2]
df = df[df["dataset"] != 3]
"""
#Project trial

df = df.sample(frac=1).reset_index(drop=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

"this model uses lowercase"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
MAX_LEN=512
TRAIN_BATCH_SIZE = 16
EPOCHS =1
LEARNING_RATE = 1e-6

class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df=df
        self.max_len = max_len
        self.text = df.text
        self.tokenizer = tokenizer
        self.targets = df["toxic?"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(text, truncation =True,
					    add_special_tokens = True, max_length = self.max_len,
					    padding="max_length", return_token_type_ids=True)
        ids = inputs["input_ids"]
        mask=inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {"ids":torch.tensor(ids,dtype=torch.long),
		"mask":torch.tensor(mask,dtype=torch.long),
		"token_type_ids":torch.tensor(token_type_ids,dtype=torch.long),
		"targets":torch.tensor(self.targets,dtype=torch.float)}


train_dataset =BERTDataset(df, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle =True, pin_memory=True)

class BERTClass(torch.nn.Module):
    
    def __init__(self):
        super(BERTClass,self).__init__()
        self.bert =BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = torch.nn.Linear(768,64)
        self.activation1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(64,1)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.bert(ids,attention_mask = mask, token_type_ids = token_type_ids, return_dict =False)
        output = self.fc1(features)
        output = self.activation1(output)
        output = self.fc2(output)
        return output

model = BERTClass()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs,targets)

optimizer = AdamW(params = model.parameters(), lr=LEARNING_RATE, weight_decay = 1e-6)

data_counter = 0
batch_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for idx, data in enumerate(train_loader,0):
        ids = data["ids"].to(device,dtype=torch.long)
        mask = data["mask"].to(device,dtype=torch.long) 
        token_type_ids = data["token_type_ids"].to(device,dtype=torch.long) 
        targets = data["targets"].to(device,dtype=torch.float)

        targets.resize_(len(targets),1)
        outputs = model(ids,mask,token_type_ids)
        loss = loss_fn(outputs, targets)
        train_loss += loss.item()
        
        data_counter += len(targets)
        if idx%100 == 0:
            print(f'Total Data: {data_counter}, loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_losses.append(train_loss / (idx + 1))


    print("The end of training...")
    print(f'Total Data: {data_counter}, loss: {loss.item()}')

model_save_path = os.getcwd() + "/bert_model"
results_path = os.getcwd() + "/results"
if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)
model.bert.save_pretrained(model_save_path)

if not os.path.isdir(results_path):
    os.makedirs(results_path)

print("Sentence Classifier loss mean is :", '%.3f' %np.mean(batch_losses))
with open(results_path+"/loss_sqd_sentclf.txt", "w") as f:
    for item in batch_losses:
        f.write("%s\n" % item)
