# -*- coding: utf-8 -*-

import os
import json
import re
from bs4 import BeautifulSoup
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
stop_words = set(stopwords.words('english'))
from tqdm import tqdm
import logging
import argparse
from logger import init_logger
import datetime


def web_crawler(input_tweet): 
    tweet = input_tweet       
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    web_text = ""
    for url in urls:
        tweet = tweet.replace(url, "")
        if url[-2:] == "/n":
            url = url[0:-2]
        raw_url = r'{}'.format(url)
        try:    
            response = requests.get(raw_url, timeout=10)
        except:
            continue
        if response.status_code == 404:
            continue
        content = response.content
        cleantext = BeautifulSoup(content, "lxml").text
        web_text = web_text + " " + cleantext
            
        #full_url = r.url
            
    full_text = tweet + " " + web_text
    return full_text


def text_processor(full_text): 
    word_tokens = word_tokenize(full_text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    stemmed_words = ""
    porter = PorterStemmer()

    for word in filtered_sentence:
        stemmed_words = stemmed_words + " " + porter.stem(word)
    #logging.info(stemmed_words)
    return stemmed_words


def process_file(input_file):
    target = []
    all_tweets = []
    with open(input_file) as json_file:
        data = json.load(json_file)

    content = list(data)
    for each in tqdm(content):
        tweet = each[0]
        input_label = each[1].strip().lower()
        
        if input_label == "lib":
            label = 1
        else:
            label = 0
            
        full_text = web_crawler(tweet)
        full_text = re.sub(r"[^a-zA-Z0-9]"," ",full_text).strip()
        #logging.info(full_text)
        processed_tweet = text_processor(full_text)
        all_tweets.append(processed_tweet)
        target.append(label)
        
    return all_tweets, target

def stratified():
    # access the data file containing the three settings in the main folder
    base_folder = '../data/stratified/'

    #base_folder = r'data\stratified'
    #base_folder = r'data\within\boto'
    #base_folder = r'data\within\docker'
    #base_folder = r'data\within\flask'
    #base_folder = r'data\within\mock'
    #base_folder = r'data\within\pandas'
    folders = os.listdir(base_folder)

    fol_counter = 1
    add_precision = 0.0
    add_recall = 0.0
    add_f1 = 0.0

    # Change folds according to the setting
    folds = 5
    
    for fol in folders:
        training_path = base_folder + str(fol_counter) + "/training.json"

        test_path = base_folder + str(fol_counter) + "/test.json"

        logging.info("Fold: " + str(fol_counter))
        
        measures = {"TP" : 0, "FP" : 0, "TN" : 0, "FN" : 0}
        X_set, Y_set = process_file(training_path)
        A_set, B_set = process_file(test_path)
        
        # Classifier and Discriminative Model
        
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_set)
        Y_train = Y_set # numpy.array(Y_set)
        
        classifier = SVC()
        classifier.fit(X_train,Y_train)
        
        X_test = vectorizer.transform(A_set)    

        Y_pred = classifier.predict(X_test)
        predictions = Y_pred

        counter = 0
        while(counter < len(B_set)):
            assigned_label = predictions[counter]
            label = B_set[counter]

            if label == 1 and assigned_label == 1:
                measures["TP"] = measures["TP"] + 1
            elif label != 1 and assigned_label == 1:
                measures["FP"] = measures["FP"] + 1
            elif label != 1 and assigned_label != 1:
                measures["TN"] = measures["TN"] + 1
            elif label == 1 and assigned_label != 1:
                measures["FN"] = measures["FN"] + 1
            counter = counter + 1
        
        try:
            precision = measures["TP"]/(measures["TP"] + measures["FP"])
        except ZeroDivisionError:
            precision = 0
        try:
            recall = measures["TP"]/(measures["TP"] + measures["FN"])
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * (precision * recall)/(precision + recall)
        except ZeroDivisionError:
            f1 = 0
        
        add_precision = add_precision + precision
        add_recall = add_recall + recall
        add_f1 = add_f1 + f1

        logging.info("Precision: " + str(precision))
        logging.info("Recall: " + str(recall))
        logging.info("F1: " + str(f1))
        
        fol_counter = fol_counter + 1
        logging.info("-------------------------------------")
        
    logging.info("---------Overall:--------------")
    logging.info("Precision: ")
    logging.info(add_precision/folds)
    logging.info("Recall: ")
    logging.info(add_recall/folds)
    logging.info("F1: ")
    logging.info(add_f1/folds)

def within_library():
    # access the data file containing the three settings in the main folder
    base_folder = '../data/within/'
    lib_names = ['boto', 'docker', 'flask', 'mock', 'pandas']

    all_precision, all_recall, all_f1 = 0, 0, 0

    for lib_name in lib_names:
        folders = os.listdir(base_folder + lib_name)

        fol_counter = 1
        add_precision = 0.0
        add_recall = 0.0
        add_f1 = 0.0

        # Change folds according to the setting
        folds = 5
        
        for fol in folders:
            training_path = base_folder + lib_name + '/' + str(fol_counter) + "/training.json"

            test_path = base_folder + lib_name + '/' + str(fol_counter) + "/test.json"

            logging.info("Fold: " + str(fol_counter))
            
            measures = {"TP" : 0, "FP" : 0, "TN" : 0, "FN" : 0}

            X_set, Y_set = process_file(training_path)
            A_set, B_set = process_file(test_path)
            
            # Classifier and Discriminative Model
            
            vectorizer = CountVectorizer()
            X_train = vectorizer.fit_transform(X_set)
            Y_train = Y_set # numpy.array(Y_set)
            
            classifier = SVC()
            classifier.fit(X_train,Y_train)
            
            X_test = vectorizer.transform(A_set)    

            Y_pred = classifier.predict(X_test)
            predictions = Y_pred

            counter = 0
            while(counter < len(B_set)):
                assigned_label = predictions[counter]
                label = B_set[counter]

                if label == 1 and assigned_label == 1:
                    measures["TP"] = measures["TP"] + 1
                elif label != 1 and assigned_label == 1:
                    measures["FP"] = measures["FP"] + 1
                elif label != 1 and assigned_label != 1:
                    measures["TN"] = measures["TN"] + 1
                elif label == 1 and assigned_label != 1:
                    measures["FN"] = measures["FN"] + 1
                counter = counter + 1
            
            try:
                precision = measures["TP"]/(measures["TP"] + measures["FP"])
            except ZeroDivisionError:
                precision = 0
            try:
                recall = measures["TP"]/(measures["TP"] + measures["FN"])
            except ZeroDivisionError:
                recall = 0
            try:
                f1 = 2 * (precision * recall)/(precision + recall)
            except ZeroDivisionError:
                f1 = 0
            
            add_precision = add_precision + precision
            add_recall = add_recall + recall
            add_f1 = add_f1 + f1

            logging.info("Precision: " + str(precision))
            logging.info("Recall: " + str(recall))
            logging.info("F1: " + str(f1))
            
            fol_counter = fol_counter + 1
            logging.info("-------------------------------------")
        
        all_precision += (add_precision / folds)
        all_recall += (add_recall / folds)
        all_f1 += (add_f1 / folds)

        logging.info("Overall:")
        logging.info("Precision: ")
        logging.info(add_precision/folds)
        logging.info("Recall: ")
        logging.info(add_recall/folds)
        logging.info("F1: ")
        logging.info(add_f1/folds)

    logging.info('------------- in total -----------------')
    logging.info('precision: {}'.format(round(all_precision / 5, 2)))
    logging.info('recall:{}'.format(round(all_recall / 5, 2)))
    logging.info('f1: {}'.format(round(all_f1 / 5, 2)))


def cross_library():
    # access the data file containing the three settings in the main folder
    base_folder = '../data/cross/'

    #base_folder = r'data\stratified'
    #base_folder = r'data\within\boto'
    #base_folder = r'data\within\docker'
    #base_folder = r'data\within\flask'
    #base_folder = r'data\within\mock'
    #base_folder = r'data\within\pandas'
    folders = os.listdir(base_folder)

    fol_counter = 1
    add_precision = 0.0
    add_recall = 0.0
    add_f1 = 0.0

    # Change folds according to the setting
    # folds = 5
    folds  = 23

    
    for fol in folders:
        training_path = base_folder + str(fol_counter) + "/training.json"

        test_path = base_folder + str(fol_counter) + "/test.json"

        logging.info("Fold: " + str(fol_counter))
        
        measures = {"TP" : 0, "FP" : 0, "TN" : 0, "FN" : 0}
        X_set, Y_set = process_file(training_path)
        A_set, B_set = process_file(test_path)
        
        # Classifier and Discriminative Model
        
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_set)
            
        Y_train = Y_set # numpy.array(Y_set)
        
        classifier = SVC()
        classifier.fit(X_train,Y_train)
        
        X_test = vectorizer.transform(A_set)    

        Y_pred = classifier.predict(X_test)
        predictions = Y_pred

        counter = 0
        while(counter < len(B_set)):
            assigned_label = predictions[counter]
            label = B_set[counter]

            if label == 1 and assigned_label == 1:
                measures["TP"] = measures["TP"] + 1
            elif label != 1 and assigned_label == 1:
                measures["FP"] = measures["FP"] + 1
            elif label != 1 and assigned_label != 1:
                measures["TN"] = measures["TN"] + 1
            elif label == 1 and assigned_label != 1:
                measures["FN"] = measures["FN"] + 1
            counter = counter + 1
        
        try:
            precision = measures["TP"]/(measures["TP"] + measures["FP"])
        except ZeroDivisionError:
            precision = 0
        try:
            recall = measures["TP"]/(measures["TP"] + measures["FN"])
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * (precision * recall)/(precision + recall)
        except ZeroDivisionError:
            f1 = 0
        
        add_precision = add_precision + precision
        add_recall = add_recall + recall
        add_f1 = add_f1 + f1

        logging.info("Precision: " + str(precision))
        logging.info("Recall: " + str(recall))
        logging.info("F1: " + str(f1))
        
        fol_counter = fol_counter + 1
        logging.info("-------------------------------------")
        
    logging.info("-----------Overall:-----------")
    logging.info("Precision: ")
    logging.info(add_precision/folds)
    logging.info("Recall: ")
    logging.info(add_recall/folds)
    logging.info("F1: ")
    logging.info(add_f1/folds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--variant', help='setting variant')
    
    args = parser.parse_args()
    
    variant = args.variant
    if variant == 'cross':
        init_logger('cross_prasetyo_{}.log'.format(datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
        logging.info('===========> cross setting <==============')
        cross_library()
    elif variant == 'stratified':
        init_logger('stratified_prasetyo_{}.log'.format(datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
        logging.info('===========> stratified setting <==============')
        stratified()
    elif variant == 'within':
        init_logger('within_prasetyo_{}.log'.format(datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
        logging.info('===========> within setting <==============')
        within_library()