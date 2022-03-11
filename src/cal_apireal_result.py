import re
import ujson

res_folder = '../res/'

def cal_stratified():
    res_file = res_folder + 'apireal_stratified_res.txt'
    with open(res_file) as f:
        file_lines = f.readlines()

    precision, recall, f1 = 0, 0, 0
    count = 0
    index = 0

    for j in range(5):
        count += 1
        pre_line, recall_line, f1_line = file_lines[index + 2 + 6 * j], \
            file_lines[index + 3 + 6 * j], file_lines[index + 4 + 6 * j]
        pl = re.findall('Precision: .+', pre_line)
        
        precision += float(re.findall("\d+\.\d+", pl[0])[0])
        rl = re.findall('Recall: .+', recall_line)

        recall += float(re.findall("\d+\.\d+", rl[0])[0])
        fl = re.findall('F1: .+', f1_line)
        
        try:
            f1 += float(re.findall("\d+\.\d+", fl[0])[0])
        except IndexError:
            f1 += float(0.0)

    print('count: {}'.format(count))
    print("precision: ", round(precision / 5, 2))
    print("recall: ", round(recall / 5, 2))
    print("f1_score: ", round(f1 / 5, 2))

def cal_within():
    res_file = res_folder + 'apireal_within_res.txt'
    with open(res_file) as f:
        file_lines = f.readlines()

    lib_names = ['boto', 'docker', 'flask', 'mock', 'pandas']

    precision, recall, f1 = 0, 0, 0
    count = 0
    for lib in lib_names:
        for index, line in zip(range(len(file_lines)), file_lines):

            lib_line = re.findall(lib, line.lower())

            if len(lib_line) == 0:
                continue
            # print(index)

            for j in range(5):
                count += 1
                pre_line, recall_line, f1_line = file_lines[index + 3 + 6 * j], \
                    file_lines[index + 4 + 6 * j], file_lines[index + 5 + 6 * j]
                pl = re.findall('Precision: .+', pre_line)
                
                precision += float(re.findall("\d+\.\d+", pl[0])[0])
                rl = re.findall('Recall: .+', recall_line)

                recall += float(re.findall("\d+\.\d+", rl[0])[0])
                fl = re.findall('F1: .+', f1_line)
                
                try:
                    f1 += float(re.findall("\d+\.\d+", fl[0])[0])
                except IndexError:
                    f1 += float(0.0)

    print('count: {}'.format(count))
    print("precision: ", round(precision / 25, 2))
    print("recall: ", round(recall / 25, 2))
    print("f1_score: ", round(f1 / 25, 2))

def cal_cross():
    res_file = res_folder + 'apireal_cross_res.txt'
    with open(res_file) as f:
        file_lines = f.readlines()

    precision, recall, f1 = 0, 0, 0
    count = 0
    index = 0

    for j in range(23):
        count += 1
        pre_line, recall_line, f1_line = file_lines[index + 2 + 6 * j], \
            file_lines[index + 3 + 6 * j], file_lines[index + 4 + 6 * j]
        pl = re.findall('Precision: .+', pre_line)
        
        try:
            precision += float(re.findall("\d+\.\d+", pl[0])[0])
        except IndexError:
            precision += float(0.0)

        rl = re.findall('Recall: .+', recall_line)

        recall += float(re.findall("\d+\.\d+", rl[0])[0])
        fl = re.findall('F1: .+', f1_line)
        
        try:
            f1 += float(re.findall("\d+\.\d+", fl[0])[0])
        except IndexError:
            f1 += float(0.0)

    print('count: {}'.format(count))
    print("precision: ", round(precision / 23, 2))
    print("recall: ", round(recall / 23, 2))
    print("f1_score: ", round(f1 / 23, 2))
    
def cal_within_single():
    res_file = res_folder + 'apireal_within_res.txt'
    with open(res_file) as f:
        file_lines = f.readlines()

    lib_names = ['boto', 'docker', 'flask', 'mock', 'pandas']

    for lib in lib_names:
        precision, recall, f1 = 0, 0, 0

        for index, line in zip(range(len(file_lines)), file_lines):

            lib_line = re.findall(lib, line.lower())

            if len(lib_line) == 0:
                continue
            # print(index)

            for j in range(5):
                pre_line, recall_line, f1_line = file_lines[index + 3 + 6 * j], \
                    file_lines[index + 4 + 6 * j], file_lines[index + 5 + 6 * j]
                pl = re.findall('Precision: .+', pre_line)
                
                precision += float(re.findall("\d+\.\d+", pl[0])[0])
                rl = re.findall('Recall: .+', recall_line)

                recall += float(re.findall("\d+\.\d+", rl[0])[0])
                fl = re.findall('F1: .+', f1_line)
                
                try:
                    f1 += float(re.findall("\d+\.\d+", fl[0])[0])
                except IndexError:
                    f1 += float(0.0)

        print(lib + '==========')
        print("precision: ", round(precision / 5, 2))
        print("recall: ", round(recall / 5, 2))
        print("f1_score: ", round(f1 / 5, 2))
        print()

def check_positive_samples_stratified():
    for i in range(1, 6):
        with open('../data/stratified/{}/test.json'.format(i), 'r') as f:
            data = ujson.load(f)
        
        positive_labels = 0

        for item in data:
            if item[1] == 'Lib':
                positive_labels += 1
        
        print(positive_labels)

if __name__ == '__main__':
    # cal_within()
    # cal_stratified()
    # check_positive_samples_stratified()
    # cal_within_single()
    cal_cross()