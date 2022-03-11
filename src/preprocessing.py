import re
import os
import codecs
import ujson
import pickle

original_data_folder = '../data'
cleaned_data_folder = '../data/cleaned'

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Replace '&amp;' with '&'
    text = re.sub(r'&[a-zA-Z0-9]+;?', "", text) # Removes words like &nbsp; , &amp; , etc
    text = re.sub('RT', "", text)               # Removes 'RT'
    text = re.sub('rt', '', text)
    text = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', "", text)    # Replaces all urls
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean(split_type, fold_num, is_train):
    if is_train:
        original_file = original_data_folder + '/{}/{}/{}.json'.format(split_type, fold_num, 'training')
    else:
        original_file = original_data_folder + '/{}/{}/{}.json'.format(split_type, fold_num, 'test')
    
    texts = []
    labels = []
    
    with codecs.open(original_file, encoding='utf-8') as f:
        data = ujson.load(f)    
        
    for item in data:
        texts.append(item[0])
        labels.append(1 if item[1] == 'Lib' else 0)
        
    cleaned_texts = list()
    for text in texts:
        cleaned_text = text_preprocessing(text)
        cleaned_texts.append(cleaned_text)
        
    # save to file
    file_prefix = 'training' if is_train else 'test'
    
    os.makedirs(cleaned_data_folder + '/{}/{}'.format(split_type, fold_num), exist_ok=True)
    
    with open(cleaned_data_folder + '/{}/{}/{}.pkl'.format(split_type, fold_num, file_prefix), 'wb') as handle:
        pickle.dump({'text': cleaned_texts, 'labels': labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    print('saved cleaned data')
    
    with open(original_data_folder + '/{}/{}/{}.pkl'.format(split_type, fold_num, file_prefix), 'wb') as handle:
        pickle.dump({'text': texts, 'labels': labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved the original pkl data')
    

def clean_lib(lib_name, fold_num, is_train):
    if is_train:
        original_file = original_data_folder + '/within/{}/{}/{}.json'.format(lib_name, fold_num, 'training')
    else:
        original_file = original_data_folder + '/within/{}/{}/{}.json'.format(lib_name, fold_num, 'test')
    
    texts = []
    labels = []
    
    with codecs.open(original_file, encoding='utf-8') as f:
        data = ujson.load(f)    
        
    for item in data:
        texts.append(item[0])
        labels.append(1 if item[1] == 'Lib' else 0)
        
    cleaned_texts = list()
    for text in texts:
        cleaned_text = text_preprocessing(text)
        cleaned_texts.append(cleaned_text)
        
    # save to file
    file_prefix = 'training' if is_train else 'test'
    
    os.makedirs(cleaned_data_folder + '/within/{}/{}'.format(lib_name, fold_num), exist_ok=True)
    
    with open(cleaned_data_folder + '/within/{}/{}/{}.pkl'.format(lib_name, fold_num, file_prefix), 'wb') as handle:
        pickle.dump({'text': cleaned_texts, 'labels': labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    print('saved cleaned data')
    
    with open(original_data_folder + '/within/{}/{}/{}.pkl'.format(lib_name, fold_num, file_prefix), 'wb') as handle:
        pickle.dump({'text': texts, 'labels': labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved the original pkl data')


if __name__ == '__main__':
    # cross
    # for split_type in ['cross']:
    #     for fold_num in range(1, 24):
    #         clean(split_type,  str(fold_num), True)
    #         clean(split_type, str(fold_num), False)

    # for lib_name in ['boto', 'docker', 'flask', 'mock', 'pandas']:
    #     for fold_num in range(1, 6):
    #         clean_lib(lib_name,  str(fold_num), True)
    #         clean_lib(lib_name, str(fold_num), False)
    split_type = 'stratified'
    
    for fold_num in range(1, 6):
        clean(split_type,  str(fold_num), True)
        clean(split_type,  str(fold_num), False)