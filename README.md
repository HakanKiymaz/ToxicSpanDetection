# Bahcesehir University
## Deep Learning Course
### 2022

*Install requirements*
```
pip install -r requirements.txt
```

| WARNING: All datasets are public except one dataset published by Golbeck, Jennifer, et al. <br/> You need to mail and sign Terms of Use agreement to achieve dataset |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
[Online harassment dataset](https://dl.acm.org/doi/10.1145/3091478.3091509)

<span style="color: blue">Project files: </span>
- helpers.py
  - Use for padding sequences.
- sqad_sentcl.py
  - Sentence classifier which uses pretrained BERT model and fine-tunes with additional data.
  - The output is another pre-trained model to be used in token classifier.
- sqad_tokclf.py
  - Token classifier of sequential adaption approach. It uses BERT tokenizer. As classifier, it adapts pre-trained model in previously mentioned model.
  - Produces f1 score as metric.
- wsl_wsl.py
  - Sentence classifier which uses BERT as both tokenizer and classifier. 
  - It produces probability of tokens being toxic.
- wsl_tokclf.py
  - Token classifier of weakly supervised approach. It uses BERT as tokenizer and previously trained model as classifier.
  - Produces f1 score as metric.

### Datasets

After obtaining data from <span style="color: blue">[Golbeck-et-al](https://dl.acm.org/doi/10.1145/3091478.3091509) </span>, please run data_process.py file in the data folder.
If you can't obtain dataset3, please modify data_process.py in order to prepare partial train dataset for sentence classifier models.

### Training steps
***Sequential adaption***

First run sqad_sentclf.py file to produce fine-tuned BERT model which will be saved in *bert_model* directory.

Then, run sqad_tokclf.py file, which uses saved model in *bert_model* directory.

***Weakly Supervised Learning***

First run wsl_wsl.py file to produce augmented BERT model. The model will be saved in _data_ folder.

Then run wsl_tokclf.py, which uses saved pickle model in data folder.

Outputs of both approaches will be saved in _results_ directory.