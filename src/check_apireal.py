import ujson

def check_labels_cross(j):
    original_folder = '../data/cross/'
    apireal_folder = '../APIReal/api_recog/cross/'

    original_data = original_folder + str(j) + '/training.json'
    apireal_data = apireal_folder + 'training{}.conll'.format(j)

    with open(original_data) as f:
        data = ujson.load(f)

    tweets = list()
    tweet_index = list()
    labels = list()

    for i, item in zip(range(len(data)), data):
        tweets.append(item[0])
        tweet_index.append(i)

        if item[1] == 'Lib':
            labels.append(1)
        else:
            labels.append(0)
    
    with open(apireal_data) as f:
        lines = f.readlines()
    
    a_labels = list()
    a_index = list()

    has_lib = 0
    count = 0
    for index in range(len(lines)):
        if len(lines[index].strip()) == 0:
            continue

        if lines[index][0] == '[' and lines[index][1] == ']':
            a_labels.append(has_lib)
            count += 1
            has_lib = 0
            a_index.append(index)
        elif lines[index].strip().split('\t')[1] == 'B-API':
            has_lib = 1

    found_error = False
    for index in range(len(a_labels)):
        if a_labels[index] != labels[index]:
            found_error = True
            print(tweets[index])
            print(tweet_index[index])
            print(a_index[index])
            print()

    if found_error:
        return 'NO'

def check_labels_stratified(j, file_type):
    original_folder = '../data/stratified/'
    apireal_folder = '../APIReal/api_recog/stratified/'

    original_data = original_folder + str(j) + '/{}.json'.format(file_type)
    apireal_data = apireal_folder + '{}{}.conll'.format(file_type, j)

    with open(original_data) as f:
        data = ujson.load(f)

    tweets = list()
    tweet_index = list()
    labels = list()

    for i, item in zip(range(len(data)), data):
        tweets.append(item[0])
        tweet_index.append(i)

        if item[1] == 'Lib':
            labels.append(1)
        else:
            labels.append(0)
    
    with open(apireal_data) as f:
        lines = f.readlines()
    
    a_labels = list()
    a_index = list()

    has_lib = 0
    count = 0
    for index in range(len(lines)):
        if len(lines[index].strip()) == 0:
            continue

        if lines[index][0] == '[' and lines[index][1] == ']':
            a_labels.append(has_lib)
            count += 1
            has_lib = 0
            a_index.append(index)
        elif lines[index].strip().split('\t')[1] == 'B-API':
            has_lib = 1

    found_error = False
    for index in range(len(a_labels)):
        if a_labels[index] != labels[index]:
            found_error = True
            print(tweets[index])
            print(tweet_index[index])
            print(a_index[index])

    if found_error:
        return 'NO'

def check_labels_within(j, lib, file_type):
    original_folder = '../data/within/{}/'.format(lib)
    apireal_folder = '../APIReal/api_recog/within/{}/'.format(lib)

    original_data = original_folder + str(j) + '/{}.json'.format(file_type)
    apireal_data = apireal_folder + '{}{}.conll'.format(file_type, j)

    with open(original_data) as f:
        data = ujson.load(f)

    tweets = list()
    tweet_index = list()
    labels = list()

    for i, item in zip(range(len(data)), data):
        tweets.append(item[0])
        tweet_index.append(i)

        if item[1] == 'Lib':
            labels.append(1)
        else:
            labels.append(0)
    
    with open(apireal_data) as f:
        lines = f.readlines()
    
    a_labels = list()
    a_index = list()

    has_lib = 0
    count = 0
    for index in range(len(lines)):
        if len(lines[index].strip()) == 0:
            continue

        if lines[index][0] == '[' and lines[index][1] == ']':
            a_labels.append(has_lib)
            count += 1
            has_lib = 0
            a_index.append(index)
        elif lines[index].strip().split('\t')[1] == 'B-API':
            has_lib = 1

    found_error = False
    for index in range(len(a_labels)):
        if a_labels[index] != labels[index]:
            found_error = True
            print(tweets[index])
            print(tweet_index[index])
            print(a_index[index])

    if found_error:
        return 'NO'

if __name__ == '__main__':
    for j in range(1, 23):
        if check_labels_cross(j) == 'NO':
            print('the error folder is {}'.format(j))

    for j in range(1, 6):
        if check_labels_stratified(j, 'training') == 'NO':
            print('the error folder is {}'.format(j))
        if check_labels_stratified(j, 'test') == 'NO':
            print('the error folder is {}'.format(j))

    # lib_names = ['boto', 'docker', 'flask', 'mock', 'pandas']

    # for lib in lib_names:
    #     for j in range(1, 6):
    #         if check_labels_within(j, lib, 'training') == 'NO':
    #             print('the error folder is {}'.format(j))
    #         if check_labels_within(j, lib, 'test') == 'NO':
    #             print('the error folder is {}'.format(j))