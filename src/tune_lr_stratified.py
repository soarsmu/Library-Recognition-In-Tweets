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


seed = 941207
seed_everything(seed)

learn_rate_list = [5e-5, 3e-5, 2e-5]
batch_size_list = [16, 32]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--variant', help='BERT variant')
    
    args = parser.parse_args()
    init_logger('../log/tune_lr_{}_{}.log'.format(args.variant, datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))


    split_dir = '../data/cleaned/stratified'

    logging.info('data folder is {}'.format(split_dir))

    logging.info('running {}'.format(args.variant))
    
    all_f1 = []
    all_precision = []
    all_recall = []
    all_accuracy = []
    
    k = 5
    for i in range(k):
        test_fold_num = str(i + 1)
        logging.info('====== training & validing on {} fold'.format(test_fold_num))
        
        os.makedirs('../model/stratified/', exist_ok=True)
        model_save_path = '../model/stratified/{}_{}_tune.pt'.format(args.variant, test_fold_num)

        ## prepare data
        with open(split_dir + '/{}/training.pkl'.format(test_fold_num), 'rb') as handle:
            train_pkl = pickle.load(handle)
        train_texts, train_labels = train_pkl['text'], train_pkl['labels']

        
        with open(split_dir + '/{}/test.pkl'.format(test_fold_num), 'rb') as handle:
            test_pkl = pickle.load(handle)
        test_texts, test_labels = test_pkl['text'], test_pkl['labels']


        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, \
            test_size=0.1, random_state=42, stratify=train_labels)
        
        tokenizer = AutoTokenizer.from_pretrained(name_tokenizer_model[args.variant], do_lower_case=True)

        # logging.info('vocab size is {}'.format(tokenizer.vocab_size))
            
        model = AutoModelForSequenceClassification.from_pretrained(name_tokenizer_model[args.variant], num_labels = 2)

        train_dataset = TweetDataset(train_texts, train_labels, name_tokenizer_model[args.variant])
        val_dataset = TweetDataset(val_texts, val_labels, name_tokenizer_model[args.variant])
        test_dataset = TweetDataset(test_texts, test_labels, name_tokenizer_model[args.variant])

        ### every fold, initialize once
        best_batch_size = 0
        best_f1 = 0
        best_lr = 0
        model_copy = 0
        best_loss = np.Inf

        setting_count = 0
        run_count = 0

        for batch_size in batch_size_list:

            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
            eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

            model.to(device)
            num_training_steps = num_epochs * len(train_dataloader)

            ## try different learning_rate
            for learning_rate in learn_rate_list:
                setting_count += 1

                logging.info('********** cur setting is {} ********'.format(setting_count))
                logging.info('batch size: {}, learning rate is {}'.format(batch_size, learning_rate))

                optimizer = AdamW(model.parameters(), lr=learning_rate)

                lr_scheduler = get_scheduler(
                    "linear",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps
                )

                progress_bar = tqdm(range(num_training_steps))

                for epoch in range(num_epochs):
                    run_count += 1
                    logging.info('{} runs so far'.format(run_count))

                    logging.info('Epoch: {}'.format(epoch + 1))

                    logging.info('===========> starting to train the model....')
                    
                    model.train()
                    for batch in train_dataloader:
                        seq, attn_masks, labels = batch
                        seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

                        outputs = model(input_ids=seq, attention_mask=attn_masks, labels=labels)
                        loss = outputs.loss
                        loss.backward()
                        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                    
                    logging.info('============> starting to evaluate the model....')

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
                    
                    logging.info('===========> the result on validation data....')
                    logging.info('accuracy: {}'.format(cur_accuracy))
                    logging.info('recall: {}'.format(cur_recall))
                    logging.info('precision: {}'.format(cur_precision))
                    logging.info('f1: {}'.format(cur_f1))
                
                    # if val_loss < best_loss:
                    #     logging.info('saving the best loss, changed from {} to {}'.format(best_loss, val_loss))
                    #     logging.info('best learning rate is {}'.format(learning_rate))
                    #     logging.info('best batch size is {}'.format(batch_size))

                    #     best_loss = val_loss
                    #     # torch.save(model.state_dict(), model_save_path)
                    #     model_copy = copy.deepcopy(model)
                    if cur_f1 >= best_f1:
                        logging.info('epoch {}'.format(epoch + 1))

                        logging.info('saving the best f1, changed from {} to {}'.format(best_f1, cur_f1))
                        logging.info('cur learning rate is {}'.format(learning_rate))
                        logging.info('cur batch size is {}'.format(batch_size))

                        best_f1 = cur_f1
                        best_lr = learning_rate
                        model_copy = copy.deepcopy(model)
                        best_batch_size = batch_size

        # logging.info('overall the best loss is {}'.format(best_loss))
        logging.info('\n=======================================================\n')
        logging.info('overall the best f1 is {}'.format(best_f1))
        logging.info('overall the best learning rate is {}'.format(best_lr))
        logging.info('overall the best batch size is {}'.format(best_batch_size))

        torch.save(model_copy.state_dict(), model_save_path)
        del loss
        torch.cuda.empty_cache()
        
        logging.info('\nloading the best performing model for {}'.format(test_fold_num))

        checkpoint = torch.load(model_save_path)
        model = AutoModelForSequenceClassification.from_pretrained(name_tokenizer_model[args.variant], num_labels = 2)

        model.load_state_dict(checkpoint)
        model.cuda()
        logging.info('.........succefully loaded')

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
        
        logging.info('\n\n------- the result on test data fold {}-------'.format(test_fold_num))
        logging.info('accuracy: {}'.format(cur_accuracy))
        logging.info('recall: {}'.format(cur_recall))
        logging.info('precision: {}'.format(cur_precision))
        logging.info('f1: {}'.format(cur_f1))
        
    average_f1 = [np.mean([x for x in all_f1])]
    average_precision = [np.mean([x for x in all_precision])]
    average_recall = [np.mean([x for x in all_recall])]
    average_accuracy = [np.mean([x for x in all_accuracy])]
    
    logging.info('\n\n==================> The averaged performance <============== ')
    logging.info('accuracy: {}'.format(average_accuracy))
    logging.info('precision: {}'.format(average_precision))
    logging.info('recall: {}'.format(average_recall))
    logging.info('f1_score: {}'.format(average_f1))