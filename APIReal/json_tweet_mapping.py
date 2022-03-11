from __future__ import division
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score


def file_reader(file_name):
    for row in open(file_name, "r"):
        yield row


def get_within_result():
    lib_names = ['boto', 'docker', 'flask', 'mock', 'pandas']

    with open('../res/apireal_within_res.txt', 'w') as f:
        print('here')
        for lib_name in lib_names:
            for fold_num in range(1, 6):
                measures = {
                    "TP" : 0,
                    "FP" : 0,
                    "TN" : 0,
                    "FN" : 0
                }

                output_file = 'api_recog/within/{}/res/output{}.data'.format(lib_name, fold_num)
                outs = file_reader(output_file)

                input_file = 'api_recog/within/{}/test{}.conll'.format(lib_name, fold_num)
                ins = file_reader(input_file)

                tweet_tokens = []
                final_label = ""
                output_labels = []
                input_labels = []

                # predicted_labels, real_labels = list(), list()

                while(True):
                    index = 0
                    try:
                        single = next(itertools.islice(ins, index, None))
                    except:
                        break
                    inps = single.strip().split("\t")
                    inp1 = inps[0]
                    inp2 = inps[-1]
                    input_labels.append(inp2)
                    out = next(itertools.islice(outs, index, None)).strip()
                    output_labels.append(out.strip())

                    if inp1 == "[]":
                        if "B-API" in output_labels:
                            final_label = "lib"    
                        else:
                            final_label = "text"

                        if "B-API" in input_labels:
                            label = "lib"    
                        else:
                            label = "text"
                        
                        tweet_tokens = []
                        input_labels = []
                        output_labels = []
                        
                        if label == "lib" and final_label == "lib":
                            measures["TP"] = measures["TP"] + 1
                        elif label != "lib" and final_label == "lib":
                            measures["FP"] = measures["FP"] + 1
                        elif label != "lib" and final_label != "lib":
                            measures["TN"] = measures["TN"] + 1
                        elif label == "lib" and final_label != "lib":
                            measures["FN"] = measures["FN"] + 1

                        # if label == 'lib':
                        #     real_labels.append(1)
                        # else:
                        #     real_labels.append(0)
                        
                        # if final_label == 'lib':
                        #     predicted_labels.append(1)
                        # else:
                        #     predicted_labels.append(0)

                    else:
                        tweet_tokens.append(inp1)

                # print(precision_score(real_labels, predicted_labels))
                # print(recall_score(real_labels, predicted_labels))
                # print(f1_score(real_labels, predicted_labels))
                precision = round(measures["TP"]/(measures["TP"] + measures["FP"]), 2)
                recall = round(measures["TP"]/(measures["TP"] + measures["FN"]), 2)
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = round(2 * (precision * recall)/(precision + recall), 2)

                if fold_num == 1:
                    f.write("{}:\n".format(lib_name))
                f.write("{}:\n".format(fold_num))
                f.write(str(measures))
                f.write('\n')
                f.write("Precision: {}\n".format(precision))
                f.write("Recall: {}\n".format(recall))
                f.write("F1: {}\n".format(f1))
                f.write('\n')

def get_stratified_result():
    with open('../res/apireal_stratified_res.txt', 'w') as f:
        for fold_num in range(1, 6):
            measures = {
                "TP" : 0,
                "FP" : 0,
                "TN" : 0,
                "FN" : 0
            }

            output_file = 'api_recog/stratified/res/output{}.data'.format(fold_num)
            outs = file_reader(output_file)

            input_file = 'api_recog/stratified/test{}.conll'.format(fold_num)
            ins = file_reader(input_file)

            tweet_tokens = []
            final_label = ""
            output_labels = []
            input_labels = []

            # predicted_labels, real_labels = list(), list()

            while(True):
                index = 0
                try:
                    single = next(itertools.islice(ins, index, None))
                except:
                    break
                inps = single.strip().split("\t")
                inp1 = inps[0]
                inp2 = inps[-1]
                input_labels.append(inp2)
                out = next(itertools.islice(outs, index, None)).strip()
                output_labels.append(out.strip())

                if inp1 == "[]":
                    if "B-API" in output_labels:
                        final_label = "lib"    
                    else:
                        final_label = "text"

                    if "B-API" in input_labels:
                        label = "lib"    
                    else:
                        label = "text"
                    
                    tweet_tokens = []
                    input_labels = []
                    output_labels = []
                    
                    if label == "lib" and final_label == "lib":
                        measures["TP"] = measures["TP"] + 1
                    elif label != "lib" and final_label == "lib":
                        measures["FP"] = measures["FP"] + 1
                    elif label != "lib" and final_label != "lib":
                        measures["TN"] = measures["TN"] + 1
                    elif label == "lib" and final_label != "lib":
                        measures["FN"] = measures["FN"] + 1

                    # if label == 'lib':
                    #     real_labels.append(1)
                    # else:
                    #     real_labels.append(0)
                    
                    # if final_label == 'lib':
                    #     predicted_labels.append(1)
                    # else:
                    #     predicted_labels.append(0)

                else:
                    tweet_tokens.append(inp1)

            # print(precision_score(real_labels, predicted_labels))
            # print(recall_score(real_labels, predicted_labels))
            # print(f1_score(real_labels, predicted_labels))
            precision = round(measures["TP"]/(measures["TP"] + measures["FP"]), 2)
            recall = round(measures["TP"]/(measures["TP"] + measures["FN"]), 2)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = round(2 * (precision * recall)/(precision + recall), 2)

            f.write("{}:\n".format(fold_num))
            f.write(str(measures))
            f.write('\n')
            f.write("Precision: {}\n".format(precision))
            f.write("Recall: {}\n".format(recall))
            f.write("F1: {}\n".format(f1))
            f.write('\n')

def get_cross_result():
    with open('../res/apireal_cross_res.txt', 'w') as f:
        for fold_num in range(1, 24):
            measures = {
                "TP" : 0,
                "FP" : 0,
                "TN" : 0,
                "FN" : 0
            }

            output_file = 'api_recog/cross/res/output{}.data'.format(fold_num)
            outs = file_reader(output_file)

            input_file = 'api_recog/cross/test{}.conll'.format(fold_num)
            ins = file_reader(input_file)

            tweet_tokens = []
            final_label = ""
            output_labels = []
            input_labels = []

            # predicted_labels, real_labels = list(), list()

            while(True):
                index = 0
                try:
                    single = next(itertools.islice(ins, index, None))
                except:
                    break
                inps = single.strip().split("\t")
                inp1 = inps[0]
                inp2 = inps[-1]
                input_labels.append(inp2)
                out = next(itertools.islice(outs, index, None)).strip()
                output_labels.append(out.strip())

                if inp1 == "[]":
                    if "B-API" in output_labels:
                        final_label = "lib"    
                    else:
                        final_label = "text"

                    if "B-API" in input_labels:
                        label = "lib"    
                    else:
                        label = "text"
                    
                    tweet_tokens = []
                    input_labels = []
                    output_labels = []
                    
                    if label == "lib" and final_label == "lib":
                        measures["TP"] = measures["TP"] + 1
                    elif label != "lib" and final_label == "lib":
                        measures["FP"] = measures["FP"] + 1
                    elif label != "lib" and final_label != "lib":
                        measures["TN"] = measures["TN"] + 1
                    elif label == "lib" and final_label != "lib":
                        measures["FN"] = measures["FN"] + 1

                    # if label == 'lib':
                    #     real_labels.append(1)
                    # else:
                    #     real_labels.append(0)
                    
                    # if final_label == 'lib':
                    #     predicted_labels.append(1)
                    # else:
                    #     predicted_labels.append(0)

                else:
                    tweet_tokens.append(inp1)

            # print(precision_score(real_labels, predicted_labels))
            # print(recall_score(real_labels, predicted_labels))
            # print(f1_score(real_labels, predicted_labels))
            
            try:
                precision = round(measures["TP"]/(measures["TP"] + measures["FP"]), 2)
            except ZeroDivisionError:
                precision = 0
                
            recall = round(measures["TP"]/(measures["TP"] + measures["FN"]), 2)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = round(2 * (precision * recall)/(precision + recall), 2)

            f.write("{}:\n".format(fold_num))
            f.write(str(measures))
            f.write('\n')
            f.write("Precision: {}\n".format(precision))
            f.write("Recall: {}\n".format(recall))
            f.write("F1: {}\n".format(f1))
            f.write('\n')

if __name__ == "__main__":
    # get_within_result()
    # get_stratified_result()
    get_cross_result()