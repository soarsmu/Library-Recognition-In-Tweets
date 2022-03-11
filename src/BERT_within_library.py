from transformers import AdamW
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from data_utils import TweetDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from transformers import get_scheduler
import copy
import pickle
from logger import init_logger
from config import *
import os

import datetime
import logging

from utils import seed_everything

import argparse

lr = 2e-5
bs = 32


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--variant', help='BERT variant')
    parser.add_argument('--lib_name', help='split type')
    parser.add_argument('--seed', help='randome seed')

    args = parser.parse_args()
    seed = int(args.seed)

    seed_everything(seed)

    variant = args.variant
    lib_name = args.lib_name


    init_logger('../log/within_{}_{}_{}.log'.format(lib_name, variant, datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))


    logging.info('learning rate is :{}, max_len is {}'.format(lr, max_len))
    logging.info('seed is {}'.format(seed))
    logging.info('batch size is {}'.format(bs))

    split_dir = '../data/cleaned/within/{}'.format(lib_name)
        
    logging.info('data folder is {}'.format(split_dir))
    logging.info('running {}'.format(variant))
    
    all_f1 = []
    all_precision = []
    all_recall = []
    all_accuracy = []
    
    k = 5
    for i in range(k):
        test_fold_num = str(i + 1)
        logging.info('====== test on {} fold'.format(test_fold_num))
        
        os.makedirs('../model/within/{}/{}'.format(lib_name, variant), exist_ok=True)
        model_save_path = '../model/within/{}_{}_{}_{}.pt'.format(lib_name, variant, test_fold_num, seed)
        
        ## prepare data
        with open(split_dir + '/{}/training.pkl'.format(test_fold_num), 'rb') as handle:
            train_pkl = pickle.load(handle)
        train_texts, train_labels = train_pkl['text'], train_pkl['labels']

        
        with open(split_dir + '/{}/test.pkl'.format(test_fold_num), 'rb') as handle:
            test_pkl = pickle.load(handle)
        test_texts, test_labels = test_pkl['text'], test_pkl['labels']


        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=seed, stratify=train_labels)
        
        tokenizer = AutoTokenizer.from_pretrained(name_tokenizer_model[variant], do_lower_case=True)

        # logging.info('vocab size is {}'.format(tokenizer.vocab_size))
        
        model = AutoModelForSequenceClassification.from_pretrained(name_tokenizer_model[variant], num_labels = 2)

        train_dataset = TweetDataset(train_texts, train_labels, name_tokenizer_model[variant])
        val_dataset = TweetDataset(val_texts, val_labels, name_tokenizer_model[variant])
        test_dataset = TweetDataset(test_texts, test_labels, name_tokenizer_model[variant])

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs)
        eval_dataloader = DataLoader(val_dataset, batch_size=bs)
        test_dataloader = DataLoader(test_dataset, batch_size=bs)

        model.to(device)
        num_training_steps = num_epochs * len(train_dataloader)

        optimizer = AdamW(model.parameters(), lr=lr)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))

        best_loss = np.Inf
        best_f1 = 0
        model_copy = 0
        
        for epoch in range(num_epochs):
            logging.info('starting to train the model....')
            
            model.train()
            for batch in train_dataloader:
                # batch = {k: v.to(device) for k, v in batch.items()}
                seq, attn_masks, labels = batch
                seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

                outputs = model(input_ids=seq, attention_mask=attn_masks, labels=labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            
            logging.info('starting to evaluate the model....')

            val_loss = []

            model.eval()
            all_predictions = []
            all_labels = []
            
            for batch in eval_dataloader:
                seq, attn_masks, labels = batch
                seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

                with torch.no_grad():
                    outputs = model(input_ids=seq, attention_mask=attn_masks, labels=labels)
                    
                val_loss.append(outputs.loss.item())
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).flatten()
                pred = predictions.cpu()
                
                truth = labels.cpu()
                all_predictions.extend(pred)
                all_labels.extend(truth)
                
            val_loss = np.mean(val_loss)
            cur_accuracy = accuracy_score(all_labels, all_predictions)
            cur_f1 = f1_score(all_labels, all_predictions)
            cur_precision = precision_score(all_labels, all_predictions)
            cur_recall = recall_score(all_labels, all_predictions)
            
            logging.info('the result on validation data....')
            logging.info('accuracy: {}'.format(cur_accuracy))
            logging.info('recall: {}'.format(cur_recall))
            logging.info('precision: {}'.format(cur_precision))
            logging.info('f1: {}'.format(cur_f1))
        
            # if val_loss < best_loss:
            #     logging.info('saving the best loss, changed from {} to {}'.format(best_loss, val_loss))
            #     best_loss = val_loss
            #     # torch.save(model.state_dict(), model_save_path)
            #     model_copy = copy.deepcopy(model)

            if cur_f1 >= best_f1:
                logging.info('saving the best f1, changed from {} to {}'.format(best_f1, cur_f1))
                best_f1 = cur_f1
                model_copy = copy.deepcopy(model)
        
        # logging.info('overall the best loss is {}'.format(best_loss))
        logging.info('overall the best f1 is {}'.format(best_f1))
        torch.save(model_copy.state_dict(), model_save_path)
        del loss
        torch.cuda.empty_cache()
        
        logging.info('loading the best performing model for {}'.format(test_fold_num))

        checkpoint = torch.load(model_save_path)
        model = AutoModelForSequenceClassification.from_pretrained(name_tokenizer_model[variant], num_labels = 2)

        model.load_state_dict(checkpoint)
        model.cuda()
        logging.info('succefully loaded')

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

        
        # cur_accuracy = np.mean(cur_accuracy)
        cur_accuracy = accuracy_score(all_labels, all_predictions)
        cur_f1 = f1_score(all_labels, all_predictions)

        cur_precision = precision_score(all_labels, all_predictions)
        cur_recall = recall_score(all_labels, all_predictions)
        
        all_f1.append(cur_f1)
        all_recall.append(cur_recall)
        all_precision.append(cur_precision)
        all_accuracy.append(cur_accuracy)
        
        logging.info('------- the result on test data -------')
        logging.info('accuracy: {}'.format(cur_accuracy))
        logging.info('recall: {}'.format(cur_recall))
        logging.info('precision: {}'.format(cur_precision))
        logging.info('f1: {}'.format(cur_f1))
        
    average_f1 = [np.mean([x for x in all_f1])]
    average_precision = [np.mean([x for x in all_precision])]
    average_recall = [np.mean([x for x in all_recall])]
    average_accuracy = [np.mean([x for x in all_accuracy])]
    
    logging.info('============ The performance across 5 fold ======== ')
    logging.info('accuracy: {}'.format(average_accuracy))
    logging.info('precision: {}'.format(average_precision))
    logging.info('recall: {}'.format(average_recall))
    logging.info('f1_score: {}'.format(average_f1))