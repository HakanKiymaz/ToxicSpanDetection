import numpy as np
import pandas as pd
import gc
import torch
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import pickle

df = pd.read_csv("./data/sentence_level_df.txt")
print("toxic sentence mean : ", df['toxic?'].mean())


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tqdm.pandas()
num_tokens = 512
encoded_info = df['text'].progress_apply(
    lambda x: tokenizer.encode_plus(x,
                                    max_length = num_tokens,
                                    padding = 'max_length',
                                    return_special_tokens_mask=True,
                                    add_special_tokens=False
                                   )).values

df['tokens'] = [x['input_ids'] for x in encoded_info]
df['masks'] = [x['special_tokens_mask'] for x in encoded_info]

df.drop(df[df['tokens'].map(len) > num_tokens].index, inplace=True)
print("toxic sentence mean after long sentence dropping : ", df['toxic?'].mean())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert = BertModel.from_pretrained('bert-base-uncased', config=config).to(device)
bert.eval()

bert_token_indices = set()
bert_token_indices_to_indices = dict()
indices_to_token = dict()

for x in df['tokens']:
    bert_token_indices = bert_token_indices.union(set(x))

idx = 0
for x in tqdm(bert_token_indices):
    bert_token_indices_to_indices[x] = idx
    indices_to_token[idx] = tokenizer.convert_ids_to_tokens([x])
    idx += 1

num_words = len(bert_token_indices)
embedding_matrix = np.zeros((num_words, 768))
with torch.no_grad():
    for bert_idx in tqdm(bert_token_indices):
        idx = bert_token_indices_to_indices[bert_idx]
        embedding_matrix[idx] = bert(torch.tensor([[bert_idx]]).to(device),
                                     torch.tensor([[1]]).to(device))[2][0][0][0].cpu().numpy()



BATCH_SIZE = 16
n_batches = df.shape[0] // BATCH_SIZE

tokens = [[bert_token_indices_to_indices[x] for x in y] for y in df['tokens'].tolist()]

train_dataset = TensorDataset(torch.tensor(np.matrix(tokens)),
                              torch.tensor(df['toxic?'].astype(int).values.tolist()),
                              torch.tensor(df['masks'].tolist()))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

del encoded_info
del tokens
del train_dataset
del tokenizer
del bert
del config
gc.collect()


class ClassifierModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.tensor(embedding_matrix),
            freeze=True)

        self.gru1 = nn.GRU(
            input_size=768,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.gru2 = nn.GRU(
            input_size=64,
            hidden_size=1,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

        self.linear1 = nn.Linear(in_features=768, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, input_ids, pad_masks, labels=None):
        w = self.embedding(input_ids)
        h = nn.Tanh()(self.linear1(w))
        p = nn.Sigmoid()(self.linear2(h)).squeeze(dim=2)

        g1 = nn.LeakyReLU(0.1)(self.gru1(w)[0])
        g2 = self.gru2(g1)[0]
        a = nn.Sigmoid()(torch.sum(g2, dim=-1))

        masks = 1 - pad_masks
        a = a * masks
        A = torch.div(a, torch.tensor(torch.sum(a, dim=1)).reshape((-1, 1)))

        Y = torch.sum(A * p, dim=1)

        loss = 0
        if labels is not None:
            loss = (
                    nn.BCELoss()(Y, labels.double()) +
                    0.01 * (nn.MSELoss()(torch.max(a, dim=1).values, labels.double()) +
                            torch.mean((torch.min(torch.where(masks == 1, a, torch.ones_like(a)), dim=1).values ** 2)))
            )
        return loss, Y, p, a


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassifierModel().double().to(device)

optimizer = AdamW(model.parameters(), lr=3e-6)
if device.type == 'gpu':
    torch.cuda.empty_cache()

EPOCHS = 5
batch_losses = []

for epoch_num in range(EPOCHS):
    model.train()
    train_loss = 0
    print("epoch", epoch_num)

    for step_num, batch_data in enumerate(tqdm(train_dataloader)):
        token_ids, labels, special_tokens = tuple(t.to(device) for t in batch_data)

        loss, _, _, _ = model(
            input_ids=token_ids,
            labels=labels,
            pad_masks=special_tokens)

        train_loss += loss.item()

        model.zero_grad()
        loss.backward()

        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(train_loss / (step_num + 1))

torch.save(model.state_dict(), 'model_wsl_wsl')

with open('./results/batch_loss_wsl_wsl.pkl', 'wb') as fh:
   pickle.dump(batch_losses, fh)


df = df[df['toxic?'] == 1]

tokens = [[bert_token_indices_to_indices[x] for x in y] for y in df['tokens'].tolist()]

train_dataset = TensorDataset(torch.tensor(np.matrix(tokens)),
                              torch.tensor(df['toxic?'].astype(int).values.tolist()),
                              torch.tensor(df['masks'].tolist()))
train_sampler = SequentialSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

del tokens
del df
del train_dataset
del train_sampler
del bert_token_indices
del bert_token_indices_to_indices
gc.collect()

test_tokens = []
test_preds = []
test_scores = []
test_Y = []
test_labels = []

model.eval()
with torch.no_grad():
    for step_num, batch_data in enumerate(tqdm(train_dataloader)):
        token_ids, labels, special_tokens = tuple(t.to(device) for t in batch_data)
        _, Y, preds, scores = model(input_ids=token_ids, pad_masks=special_tokens)
        for sample_tokens, sample_Y, sample_preds, sample_scores, sample_label, sample_special_tokens in zip(
                token_ids.tolist(),
                Y.tolist(),
                preds.tolist(),
                scores.tolist(),
                labels.tolist(),
                special_tokens):
            l = 512 - sample_special_tokens.sum()
            test_tokens.append(sample_tokens[:l])
            test_Y.append(sample_Y)
            test_preds.append(sample_preds[:l])
            test_scores.append(sample_scores[:l])
            test_labels.append(sample_label)

del model
del train_dataloader
gc.collect()

df = pd.DataFrame(
    {
        'tokens': [[indices_to_token[x][0] for x in y] for y in test_tokens],
        'Y': test_Y,
        'scores': test_scores,
        'preds': test_preds,
        'label': test_labels
    }
)

df.to_pickle('./data/df_results_wsl_wsl.pkl')
