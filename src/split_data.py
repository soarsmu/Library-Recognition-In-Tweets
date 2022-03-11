from sklearn.model_selection import StratifiedKFold
import json
import ujson
import os
import numpy
import random
random.seed(42)
import collections

folder_path = '../data/agreed/'

files = os.listdir(folder_path)

def split_mixed():
    output_path = '../data/stratified/'
    feature_set = []
    target = []
    counter = 1

    for file in files:
        file_path = folder_path + "/" + file
        with open(file_path) as json_file:
            data = json.load(json_file)
        
        for key, value in data.items():
            feature_set.append(key)
            target.append(value)
        
    X = numpy.array(feature_set)
    y = numpy.array(target)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # X is the feature set and y is the target
    for train_index, val_index in skf.split(X,y): 
        #print("Train:", train_index, "Validation:", val_index) 
        #print(len(train_index), len(val_index))
        X_train, X_test = X[train_index], X[val_index] 
        y_train, y_test = y[train_index], y[val_index]
        
        training_data = []
        test_data = []
        
        for a, b in zip(X_train, y_train):
            training_data.append((a, b))
            
        for a, b in zip(X_test, y_test):
            test_data.append((a, b))
        
        os.makedirs(output_path + str(counter), exist_ok=True)
        
        training_path = output_path + str(counter) + "/training.json"
        test_path = output_path + str(counter) + "/test.json"
        
        with open(training_path, 'w') as f:
            json.dump(training_data, f, indent=4)
            
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=4)
            
        counter = counter + 1


def split_within_lib():
    folder_path = '../data/agreed/'
    library_names = set(['boto', 'docker', 'flask', 'mock', 'pandas'])
    
    files = os.listdir(folder_path)
    output_path = '../data/within/'
    os.makedirs(output_path, exist_ok=True)

    for file in files:
        if not file.split('.')[0] in library_names:
            continue
        
        counter = 1
        feature_set = []
        target = []
        file_path = folder_path + "/" + file
        file_name = file.split('.')[0]
        
        with open(file_path) as json_file:
            data = json.load(json_file)
            
        for key, value in data.items():
            feature_set.append(key)
            target.append(value)
        
        X = numpy.array(feature_set)
        y = numpy.array(target)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # X is the feature set and y is the target
        for train_index, val_index in skf.split(X, y):
            #print("Train:", train_index, "Validation:", val_index) 
            #print(len(train_index), len(val_index))
            
            X_train, X_test = X[train_index], X[val_index] 
            y_train, y_test = y[train_index], y[val_index]
            
            # 4 training parts, 1 test part
            
            training_data = []
            test_data = []
            
            for a, b in zip(X_train, y_train):
                training_data.append((a, b))
                
            for a, b in zip(X_test, y_test):
                test_data.append((a, b))
            
            os.makedirs(output_path + file_name + "/" + str(counter), exist_ok=True)
            
            training_path = output_path + file_name + "/" + str(counter) + "/training.json"
            test_path = output_path + file_name + "/" + str(counter) + "/test.json"
        
            with open(training_path, 'w') as f:
                json.dump(training_data, f, indent=4)
            
            with open(test_path, 'w') as f:
                json.dump(test_data, f, indent=4)
            
            counter = counter + 1

def cross_lib():
    folder_path = '../data/agreed/'

    files = os.listdir(folder_path)
    output_path = '../data/cross/'
    os.makedirs(output_path, exist_ok=True)
    counter = 1
    
    for i in range(23):
        test_lib = files[i]
        print(test_lib)
        
        train_libs = []
        
        for j in range(23):
            if j != i:
                train_libs.append(files[j])
                
        # print('train: {}'.format(train_libs))
        # print('test: {}'.format(test_lib))
        
        train_feature_set = []
        train_target = []
        test_feature_set = []
        test_target = []
        
        
        # for training
        for file in train_libs:
            file_path = folder_path + file
            
            with open(file_path) as json_file:
                data = json.load(json_file)
                
            for key, value in data.items():
                train_feature_set.append(key)
                train_target.append(value)
    
        X_train = numpy.array(train_feature_set)
        y_train = numpy.array(train_target)
        
        # for test

        test_file_path = folder_path + test_lib
        with open(test_file_path) as json_file:
            data = json.load(json_file)
            
        for key, value in data.items():
            test_feature_set.append(key)
            test_target.append(value)
            
        X_test = numpy.array(test_feature_set)
        y_test = numpy.array(test_target)
        
        os.makedirs(output_path + str(counter), exist_ok=True)
        training_path = output_path + str(counter) + "/training.json"
        test_path = output_path + str(counter) + "/test.json"
        
        counter = counter + 1
        
        training_data = []
        test_data = []
        
        index_shuf = list(range(len(train_feature_set)))
        random.shuffle(index_shuf)
        
        for index in index_shuf:
            training_data.append((X_train[index], y_train[index]))
        
            # for a, b in zip(X_train, y_train):
            #     training_data.append((a, b))
            
        for a, b in zip(X_test, y_test):
            test_data.append((a, b))
        
        with open(training_path, 'w') as f:
            json.dump(training_data, f, indent=4)
            
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=4)
            
        
def check_data_within():
    # mixed_folder = '../data/stratified/'
    within_folder = '../data/within/'
    library_names = ['boto', 'docker', 'flask', 'mock', 'pandas']
    
    for lib_name in library_names:
        for i in range(1, 6):
            for file_type in ['training.json', 'test.json']:
                print(lib_name + " " + file_type)
                pos, neg = 0, 0
                with open(within_folder + lib_name + "/" + str(i) + "/" + file_type) as f:
                    data = ujson.load(f)
                contents = list(data)
                for j in range(len(contents)):
                    if contents[j][1] == 'Lib':
                        pos += 1
                    else:
                        neg += 1
                print('pos/total: {}'.format(pos / (pos + neg)))

def check_data_cross(is_train):
    cross_folder = '../data/cross/'
    lib_list = ['bcrypt.json', 'packaging.json', 'flask.json', \
                'google protobuf.json', 'requests.json', 'pluggy.json', \
                'absl.json', 'colorama.json', 'more itertools.json',  \
                'docker.json', 'mock.json', 'apiclient.json', \
                'cryptography.json', 'prometheus client.json',  'appdirs.json', \
                'prompt toolkit.json', 'google auth httplib2.json', 'pandas.json', \
                'futures.json', 'ipaddress.json', 'websocket.json', \
                'click.json', 'boto.json']
    
    if is_train:
        for i in range(1, 23):
            for file_type in ['test.json']:
                lib_name = lib_list[i - 1]
                print(lib_name + " " + file_type)
                
                pos, neg = 0, 0
                with open(cross_folder + str(i) + "/" + file_type) as f:
                    data = ujson.load(f)
                    
                contents = list(data)
                for j in range(len(contents)):
                    if contents[j][1] == 'Lib':
                        pos += 1
                    else:
                        neg += 1
                        
                print('pos / total: {}'.format(pos / (pos + neg)))
                print('pos: {}'.format(pos))
                print('neg: {}'.format(neg))
                
                ori_pos, ori_neg = 0, 0
                with open('../data/agreed/{}'.format(lib_list[i - 1])) as f:
                    data = ujson.load(f)
                    
                for key, value in data.items():
                    if value == 'Lib':
                        ori_pos += 1
                    else:
                        ori_neg += 1
                        
                print('ori_pos / total: {}'.format(ori_pos / (ori_pos + ori_neg)))
                print('ori_pos: {}'.format(ori_pos))
                print('ori_neg: {}'.format(ori_neg))
                
                if pos != ori_pos or neg != ori_neg:
                    print('error' + '!' * 20)
    else:
        # check test
        for i in range(1, 23):
            for file_type in ['training.json']:
                lib_name = lib_list[i - 1]
                print(lib_name + " " + file_type)
                
                train_dict = dict()
                
                with open(cross_folder + str(i) + "/" + file_type) as f:
                    data = ujson.load(f)
                    
                contents = list(data)
                for j in range(len(contents)):
                    train_dict[contents[j][0]] = contents[j][1]
                
                collections.OrderedDict(sorted(train_dict.items()))
                
                ori_train_dict = dict()
                for k in range(23):
                    if k != (i - 1):
                        with open('../data/agreed/{}'.format(lib_list[k])) as f:
                            data = ujson.load(f)
                        
                        for key, value in data.items():
                            ori_train_dict[key] = value
                
                collections.OrderedDict(sorted(ori_train_dict.items()))
                
                if train_dict != ori_train_dict:
                    print('error' + '!' * 20)

if __name__ == '__main__':
    # split_mixed()
    # split_within_lib()
    # cross_lib()
    # check_data_within()
    check_data_cross(False)