import torch
from utils import seed_everything
from transformers import AutoModelForSequenceClassification
from data_utils import TweetDataset
from torch.utils.data import DataLoader
import pandas as pd
from config import *
import pickle
import argparse

bs = 32
seed = 42
seed_everything(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--variant', help='BERT variant')
    parser.add_argument('--fold_num', help='fold num')
    
    args = parser.parse_args()
    variant = args.variant
    fold_num = int(args.fold_num)
    split_dir = '../data/cleaned/stratified/{}'.format(fold_num)
    model_save_path = '../model/stratified/{}_{}_42.pt'.format(variant, fold_num)

    checkpoint = torch.load(model_save_path)
    model = AutoModelForSequenceClassification.from_pretrained(name_tokenizer_model[variant], num_labels = 2)

    model.load_state_dict(checkpoint)
    model.cuda()

    with open(split_dir + '/test.pkl', 'rb') as handle:
        test_pkl = pickle.load(handle)
    test_texts, test_labels = test_pkl['text'], test_pkl['labels']


    test_dataset = TweetDataset(test_texts, test_labels, name_tokenizer_model[variant])
    test_dataloader = DataLoader(test_dataset, batch_size=bs)

    model.eval()
    cur_accuracy = []
    test_loss = []
    cur_precision = []
    cur_recall = []
    cur_f1 = []
    all_predictions, all_labels = [], []

    for batch in test_dataloader:
        seq, attn_masks, labels = batch
        seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=seq, attention_mask=attn_masks, labels=labels)
            
        test_loss.append(outputs.loss.item())
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).flatten()

        pred = predictions.cpu()
        truth = labels.cpu()
        all_predictions.extend(pred)
        all_labels.extend(truth)

    all_predictions = [x.item() for x in all_predictions]
    all_labels = [x.item() for x in all_labels]
    predicted_df = pd.DataFrame({'tweet': test_texts, 'label': all_labels, 'predicted': all_predictions})
    predicted_df.to_csv('stratified_{}_{}.csv'.format(variant, fold_num))