"""
Get and save the agreed tweets from two annotators
& calculate Inter-rater reliability, Cohen's Kappa

"""

import ujson
import os
import pprint
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)

data_folder = '../data/'

pos_count, neg_count = 0, 0

pos_tweets = list()

def get_agreed_tweets():
    annotator_1 = data_folder + 'excel/'
    annotator_2 = data_folder + 'annotator_2/'
    total_disagree = 0
    total_agree = 0
    
    # from annotater_1
    for file in os.listdir(annotator_1):
        if os.path.isdir(annotator_1 + file):
            continue

        library_name = file.split('.')[0]
        
        # with open(annotator_1 + file) as f:
        #     cur_file = ujson.load(f)
        #     for line in cur_file:
        #         label_1.append(line[1])
        
        df1 = pd.read_excel(annotator_1 + "{}.xlsx".format(library_name), \
                        engine='openpyxl', header=None)
        df1.columns = ['tweet', 'label']
        df1.label.apply(str)
        df1.tweet.apply(str)
        
        # with open('../data/unlabeled/{}.json'.format(file.split('.')[0]), 'w') as file_handler:
        #     ujson.dump(tweet_dict, file_handler, indent = 6)
        
        # from annotator_2
        if not os.path.exists(annotator_2 + "{}.xlsx".format(library_name)):
            continue
        
        df = pd.read_excel(annotator_2 + "{}.xlsx".format(library_name), \
                        engine='openpyxl', header=None)
        df.columns = ['tweet', 'label']
        df.label.apply(str)
        df.tweet.apply(str)
        
        disagreed = 0
        agreed = 0
        # agreed_tweets = list()
        # agreed_labels = list()
        agreed_dict = dict()
        
        if df1.shape != df.shape:
            print(library_name)
            
        for index, row in df.iterrows():
            if row['label'].lower() != df1.iloc[index]['label'].lower():
                disagreed += 1
            else:
                
                if row['tweet'] in agreed_dict:
                    print(row['tweet'])
                    continue
                agreed += 1
                agreed_dict[row['tweet']] = row['label']
                # agreed_tweets.append(row['tweet'])
                # agreed_labels.append(row['label'])
        
        os.makedirs(data_folder + 'agreed_2', exist_ok=True)
        
        with open(data_folder + 'agreed_2/{}.json'.format(library_name), 'w') as json_file:
            # for i in range(len(agreed_labels)):
            #     json_file.write(ujson.dumps({agreed_tweets[i]: agreed_labels[i]}))
            #     json_file.write('\n')
            ujson.dump(agreed_dict, json_file, indent=4)
        
        # print('>----------------<')
        # print('library: {}'.format(library_name))
        # print("disagreed: {}".format(disagreed))
        # print("agreed: {}".format(agreed))
        total_disagree += disagreed
        total_agree += agreed
        
    print('total agreed: {}'.format(total_agree))
    print('total disagreed: {}'.format(total_disagree))
    print(total_agree / (total_agree + total_disagree))
    
    
def final_data_statistics():
    agreed_folder = data_folder + 'agreed/'
    total_pos, total_neg = 0, 0
    lib_count = 0
    
    for file in sorted(os.listdir(agreed_folder)):
        if os.path.isdir(agreed_folder + file):
            continue

        tail = file.split('.')[1]
        library_name = file.split('.')[0]
        
        pos, neg = 0, 0
        with open(agreed_folder + file) as f:
            cur_file = ujson.load(f)
            
            for key, value in cur_file.items():
            # for line in cur_file:
                if value == 'Lib':
                    pos += 1
                else:
                    neg += 1
        
        # break
        total_pos += pos
        total_neg += neg
        print('>---------------<')
        print('libray: {}'.format(library_name))
        print('positive: {}'.format(pos))
        print('negative: {}'.format(neg))
        print('total: ', pos + neg)

        if pos == 0:
            print('!!!!!!!!!!!!!!'*2)
            print(library_name)
        lib_count += 1
    
    print('total lib', lib_count)
    print('total pos: {}'.format(total_pos))
    print('total neg: {}'.format(total_neg))
    print('total: {}'.format(total_neg + total_pos))


def generate_agreed():
    annotator_1 = data_folder + 'excel/'
    annotator_2 = data_folder + 'annotator_2/'
    
    files = os.listdir(annotator_1)

    lib = 0
    text = 0

    for index, file in enumerate(files):
        xl = pd.ExcelFile(annotator_2 + file, engine='openpyxl')
        df = xl.parse("Sheet", header=None, names=['Tweets', 'label'])
        ## print(df.head())
        # print(file)
        # print(df['label'].value_counts())
        try:
            lib = lib + df['label'].value_counts().Lib
        except:
            lib = lib + 0
        try:
            text = text + df['label'].value_counts().Text
        except:
            text = text + 0
    
    print('hadi')
    print('Lib:', lib)
    print('Text:', text)
    print('Total:', text + lib)
    
    agreed_lib = 0
    agreed_text = 0

    for index, file in enumerate(files):
        xl1 = pd.ExcelFile(annotator_2 + file, engine='openpyxl')
        df1 = xl1.parse("Sheet", header=None, names=['Tweets', 'label'])
        xl2 = pd.ExcelFile(annotator_1 + file, engine='openpyxl')
        df2 = xl2.parse("Sheet", header=None, names=['Tweets', 'label'])
        
        df = pd.DataFrame()
        df = df1.merge(df2, how = 'inner', indicator=False)
        df.drop_duplicates(subset=['Tweets'], inplace=True)
        ## print(df['label'].value_counts())
        
        try:
            agreed_lib = agreed_lib + df['label'].value_counts().Lib
        except:
            agreed_lib = agreed_lib + 0
        try:
            agreed_text = agreed_text + df['label'].value_counts().Text
        except:
            agreed_text = agreed_text + 0
        
        agreed_tweets = dict()
        
        for index, row in df.iterrows():
            agreed_tweets[row['Tweets']] = row['label']
        
        # with open(data_folder + 'agreed_1/{}.json'.format(file.split('.')[0]), 'w') as json_file:
        #     ujson.dump(agreed_tweets, json_file, indent=4)
            
        ## print(file, ':', agreed_lib, agreed_text)
        
    print('agreed Lib:', agreed_lib)
    print('agreed Text:', agreed_text)
    print('agreed Total:', agreed_text + agreed_lib)

def check_diff():
    agreed_1 = '../data/agreed_1/'
    agreed_2 = '../data/agreed_2/'
    
    for file in os.listdir(agreed_1):
        lib_nam = file.split('.')[0]
        with open(agreed_1 + file) as f:
            dict1 = ujson.load(f)
        
        with open(agreed_2 + file) as f:
            dict2 = ujson.load(f)
            
        z = set(dict1.keys()).difference(set(dict2.keys()))
        print(file)
        print(z)
        
if __name__ == '__main__':
    # get_agreed_tweets()
    final_data_statistics()
    # generate_agreed()
    # check_diff()
    
    # agreed_lib = 1089
    # agreed_text = 4842
    # total = 6161 

    # IRR = (agreed_lib + agreed_text) / total
    # print('IRR: ', IRR)
    
    # hadi_lib = 1282
    # divya_lib = 1125
    # hadi_text = 4879
    # divya_text = 5036
    # p0 = IRR
    
    # pE = ((hadi_lib/total)*(divya_lib/total)) + ((hadi_text/total)*(divya_text/total))
    # cohens_kappa = 1-((1-p0)/(1-pE))
    # print(cohens_kappa)