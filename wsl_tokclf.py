import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import torch
import spacy
import pickle
from bisect import bisect_left, bisect_right
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import BertTokenizer, BertModel
from helpers import pad_sequences, text_preprocess, plot_seq_len
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from IPython.display import clear_output

data = pd.read_csv("./data/tsd_train.txt")
data_test = pd.read_csv("./data/tsd_test.txt")
df_zst = pd.read_pickle('./data/df_results_wsl_wsl.pkl')
zst_scores = []
for _, x in df_zst.iterrows():
    zst_scores += x['preds']

zst_tokens, zst_labels = [], []
for _, x in df_zst.iterrows():
    if x['Y'] >= 0.5:
        zst_tokens.append(x['tokens'])
        zst_labels.append(x['preds'])

data_zst = pd.DataFrame({'bert_tokens': zst_tokens, 'token_labels': zst_labels})


def clean_spans(df):
    for index, sample in df.iterrows():
        spans = eval(sample['spans'])
        correct_spans = spans.copy()
        chars = list(sample['text'])
        for i, char in enumerate(chars):
            if i == 0:
                continue
            if (i in spans) and (i - 1 not in spans) and (chars[i - 1].isalnum()) and (char.isalnum()):
                correct_spans.append(i - 1)
            elif (i - 1 in spans) and (i not in spans) and (chars[i - 1].isalnum()) and (char.isalnum()):
                correct_spans.append(i)
        correct_spans.sort()
        sample['spans'] = correct_spans
    return df


data = clean_spans(data)
print("clean spans is applied to data")

def get_toxic_tokens(df):
    df['toxic_tokens'] = [list() for x in range(len(df.index))]
    for _, sample in df.iterrows():
        toxic = ''
        for i, char in enumerate(list(sample["text"])):
            if i in sample["spans"]:
                toxic += char
            elif len(toxic):
                sample['toxic_tokens'].append(toxic)
                toxic = ''
        if toxic:  # added to take care of the last toxic token in text
            sample['toxic_tokens'].append(toxic)

    return df


data = get_toxic_tokens(data)
nlp = spacy.blank("en")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def create_token_labels(x):
    text = nlp(x['text'])
    token_start = [token.idx for token in text]
    token_end = [token.idx + len(token) - 1 for token in text]
    toxic_ranges = ranges(x['spans'])
    l = len(x['text'])
    for range in toxic_ranges:
        start, end = range
        if end >= l:
            end = l - 1
        while start < l and x['text'][start] == ' ':
            start += 1
        while end >= 0 and x['text'][end] == ' ':
            end -= 1
        start = token_start[bisect_right(token_start, start) - 1]
        end = token_end[bisect_left(token_end, end)]
        if start >= end:
            print('Error:', x['text'])
            continue
        token_span = text.char_span(start, end + 1)
        for token in token_span:
            token.ent_type_ = 'toxic'

    bert_tokens = []
    token_labels = []
    for token in text:
        bert_subtokens = tokenizer.tokenize(token.text)
        bert_tokens += bert_subtokens
        token_labels += [int(token.ent_type_ == 'toxic') for _ in bert_subtokens]

    return bert_tokens, token_labels


def get_bert_tokens(df):
    df['bert_tokens'] = [list() for x in range(len(df.index))]
    df['token_labels'] = [list() for x in range(len(df.index))]

    for _, sample in df.iterrows():
        sample['bert_tokens'], sample['token_labels'] = create_token_labels(sample)

    return df


data = get_bert_tokens(data)
print("bert tokens ready")

train_labels_sum, zst_labels_sum, train_labels_n, zst_labels_n = 0, 0, 0, 0
for _, x in data.iterrows():
    train_labels_sum += sum(x['token_labels'])
    train_labels_n += len(x['token_labels'])
for _, x in df_zst.iterrows():
    zst_labels_sum += sum(x['preds'])
    zst_labels_n += len(x['preds'])
mean_diff = zst_labels_sum / zst_labels_n - train_labels_sum / train_labels_n

zst_labels = [[max(0, x - mean_diff) for x in y] for y in zst_labels]

data_test['bert_tokens'] = [list() for x in range(len(data_test.index))]

for _, sample in data_test.iterrows():
    text = nlp(sample['text'])
    for token in text:
        sample['bert_tokens'] += tokenizer.tokenize(token.text)


f2 = plot_seq_len(data, name='train')
f2.savefig("./results/train_seq_len_wsl_tokclf.png")
maxlen_train = max([len(x) for x in data['bert_tokens']])
print("maxlen_train : ", maxlen_train)

f3 = plot_seq_len(data_test, name='test')
f3.savefig("./results/test_seq_len_wsl_tokclf.png")
maxlen_test = max([len(x) for x in data_test['bert_tokens']])
print("maxlen_test : ", maxlen_test)

f4 = plot_seq_len(data_zst, name='zst')
f4.savefig("./results/zst_seq_len_wsl_tokclf.png")
maxlen_zst = max([len(x) for x in data_zst['bert_tokens']])
print(maxlen_zst)

maxlen = max(max(maxlen_train, maxlen_test), maxlen_zst)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_tokens = list(map(lambda t: ['[CLS]'] + t[:maxlen - 2] + ['[SEP]'], data['bert_tokens']))
test_tokens = list(map(lambda t: ['[CLS]'] + t[:maxlen - 2] + ['[SEP]'], data_test['bert_tokens']))
zst_tokens = list(map(lambda t: ['[CLS]'] + t[:maxlen - 2] + ['[SEP]'], data_zst['bert_tokens']))

print("tokens created with special tokens")
print(len(train_tokens), len(test_tokens), len(zst_tokens))
def pad_tokens(tokens, max_len=maxlen):
    tokens_len = len(tokens)
    pad_len = max(0, max_len - tokens_len)
    return (
        pad_sequences([tokens], maxlen=max_len, truncating="post", padding="post", dtype="int"),
        np.concatenate([np.ones(tokens_len, dtype="int"), np.zeros(pad_len, dtype="int")], axis=0)
    )


def get_token_ids_and_masks(tokens):
    token_ids, masks = [], []

    for x in tokens:
        token_id, mask = pad_tokens(tokenizer.convert_tokens_to_ids(x))
        token_ids.append(token_id[0])
        masks.append(mask)

    token_ids = np.array(token_ids)
    masks = np.array(masks)

    return token_ids, masks


train_token_ids, train_masks = get_token_ids_and_masks(train_tokens)
test_token_ids, test_masks = get_token_ids_and_masks(test_tokens)
zst_token_ids, zst_masks = get_token_ids_and_masks(zst_tokens)

train_token_labels = list(map(lambda t: [0] + t[:maxlen - 2] + [0], data['token_labels']))
zst_token_labels = list(map(lambda t: [0] + t[:maxlen - 2] + [0], zst_labels))

train_y = pad_sequences(train_token_labels, maxlen=maxlen, truncating="post", padding="post")[:, :, None]
zst_y = pad_sequences(zst_token_labels, maxlen=maxlen, truncating="post", padding="post", dtype='float32')[:, :, None]
train_y.shape, zst_y.shape, np.mean(train_y), np.mean(zst_y)


class BertClassifier(nn.Module):

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.hidden = nn.Linear(bert.config.hidden_size, 64)
        self.hidden_activation = nn.LeakyReLU(0.1)
        self.output = nn.Linear(64, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs[0]
        cls_output = self.hidden(cls_output)
        cls_output = self.hidden_activation(cls_output)
        cls_output = self.output(cls_output)
        cls_output = self.output_activation(cls_output)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels.float())
        return loss, cls_output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertClassifier(BertModel.from_pretrained('bert-base-uncased')).to(device)
print("BERT initialized")

BATCH_SIZE = 16

train_dataset = TensorDataset(
    torch.cat((torch.tensor(zst_token_ids), torch.tensor(train_token_ids)), 0),
    torch.cat((torch.tensor(zst_masks), torch.tensor(train_masks)), 0),
    torch.cat((torch.tensor(zst_y),torch.tensor(train_y)), 0))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(torch.tensor(test_token_ids), torch.tensor(test_masks))
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

optimizer = Adam(model.parameters(), lr=3e-6)
torch.cuda.empty_cache()

EPOCHS = 5
loss = nn.BCELoss()
total_len = len(train_token_ids) + len(zst_token_ids)
batch_losses = []

for epoch_num in range(EPOCHS):
    model.train()
    for step_num, batch_data in enumerate(tqdm(train_dataloader)):
        token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

        loss, _ = model(input_ids=token_ids, attention_mask=masks, labels=labels)

        train_loss = loss.item()

        model.zero_grad()
        loss.backward()

        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_losses.append(train_loss)


with open('./results/loss_wsl_tokclf.pkl', 'wb') as f:
    pickle.dump(batch_losses, f)

model.eval()

test_tokens = []
test_attention_masks = []
test_preds = []

with torch.no_grad():
    for step_num, batch_data in enumerate(tqdm(test_dataloader)):
        token_ids, masks = tuple(t.to(device) for t in batch_data)
        _, output = model(input_ids=token_ids, attention_mask=masks)
        test_tokens += token_ids.tolist()
        test_attention_masks += masks.tolist()
        test_preds += output[:, :, 0].tolist()

def get_metrics(predictions, gold, n):

  predictions = set(predictions)
  gold = set(gold)
  if len(gold) == 0:
      return {
          'f1': 1 if len(predictions) == 0 else 0,
          'tpr': 1 if len(predictions) == 0 else 0,
          'fpr': 0

      }
  nom = 2 * len(predictions.intersection(gold))
  denom = len(predictions) + len(gold)
  if len(gold) == n:
      return {
          'f1': nom / denom,
          'tpr': len(predictions.intersection(gold)) / len(gold),
          'fpr': 0 if n == len(gold) and len(predictions) == n else 1
      }
  return {
      'f1': nom / denom,
      'tpr': len(predictions.intersection(gold)) / len(gold),
      'fpr': len(predictions.difference(gold)) / (n - len(gold))
  }


def get_scores(threshold):
  n = len(test_tokens)
  y_pred = []
  all_spans = []
  f1, tpr, fpr = 0, 0, 0
  for i in range(n):
      reconstructed_text = ''
      spans = []
      original_text = text_preprocess(data_test.iloc[i]['text'])
      original_l = len(original_text)
      idx = 0
      tokens = tokenizer.convert_ids_to_tokens(test_tokens[i])
      n_tokens = len(tokens)
      j = 1
      last_token = None
      prev_token_toxic = False
      prev_token_prob = 0
      prev_idx = -1
      while j < n_tokens:
          while idx < original_l and original_text[idx].isspace():
              reconstructed_text += original_text[idx]
              idx += 1
          word = tokens[j]
          if word == '[SEP]':
              last_token = j - 1
              break
          if ord(original_text[idx]) == 65039:
              print('Problematic char at', idx, i)
              reconstructed_text += original_text[idx]
              idx += 1
          if word == '[UNK]':
              reconstructed_text += original_text[idx]
              idx += 1
              j += 1
              continue
          max_toxic_prob = test_preds[i][j]
          while j < n_tokens - 1 and tokens[j + 1].startswith('##'):
              j += 1
              word += tokens[j][2:]
              max_toxic_prob = max(max_toxic_prob, test_preds[i][j])
          word_l = len(word)
          y_pred += [min(prev_token_prob, max_toxic_prob) for _ in range(prev_idx + 1, idx)]
          if min(prev_token_prob, max_toxic_prob) >= threshold:
              spans += list(range(prev_idx + 1, idx))
          y_pred += [max_toxic_prob for _ in range(word_l)]
          is_toxic = (max_toxic_prob >= threshold)
          prev_token_prob = max_toxic_prob
          if word == original_text[idx: idx + word_l].lower():
              reconstructed_text += word
              if is_toxic:
                  spans += list(range(idx, idx + word_l))
              idx += word_l
              prev_idx = idx - 1
              prev_token_toxic = is_toxic
          else:
              print(word, original_text[idx: idx + word_l])
          j += 1

      while idx < original_l and original_text[idx].isspace():
          reconstructed_text += original_text[idx]
          idx += 1
      y_pred += [prev_token_prob for _ in range(prev_idx + 1, idx)]

      if reconstructed_text != original_text.lower():
          print('ISSUE', i)
      else:
          metrics = get_metrics(spans, eval(data_test.iloc[i]['spans']), len(original_text))
          f1 += metrics['f1']
          tpr += metrics['tpr']
          fpr += metrics['fpr']

      all_spans.append(spans)

  return f1 / n, tpr / n, fpr / n, all_spans

thresholds = [0.01 * x for x in range(100)]
f1_scores, tpr_scores, fpr_scores = [], [], []
for x in tqdm(thresholds):
    scores = get_scores(x)
    f1_scores.append(scores[0])
    tpr_scores.append(scores[1])
    fpr_scores.append(scores[2])


with open('./results/spans_pred_wsl_tokclf.txt', 'w') as out:
    for uid, text_scores in zip(range(len(test_tokens)), get_scores(0.5)[3]):
        out.write(f'{str(uid)}\t{str(text_scores)}\n')


with open('./results/f1_wsl_tokclf.pkl', 'wb') as f:
    pickle.dump(f1_scores, f)
with open('./results/tpr_wsl_tokclf.pkl', 'wb') as f:
    pickle.dump(tpr_scores, f)
with open('./results/fpr_wsl_tokclf.pkl', 'wb') as f:
    pickle.dump(fpr_scores, f)
