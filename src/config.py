import torch

max_len = 100
num_epochs = 10

name_tokenizer_model = {
    'BERT': '../PTM/bert-base-uncased',
    'BERTOverflow': '../PTM/BERTOverflow',
    'BERTweet': '../PTM/bertweet-base',
    'RoBERTa': '../PTM/roberta-base',
    'xlnet': '../PTM/xlnet-base-cased'
}


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")