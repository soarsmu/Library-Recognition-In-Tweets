import os
import json

def mixed_library():
    base_folder = '../data/stratified/'

    folders = os.listdir(base_folder)
    fol_counter = 1
    add_precision = 0.0
    add_recall = 0.0
    add_f1 = 0.0
    folds = 5

    for fol in folders:
        fol_path = base_folder + str(fol_counter) + "/test.json"

        print("Fold: " + str(fol_counter))

        with open(fol_path) as json_file:
            data = json.load(json_file)

        measures = {"TP" : 0, "FP" : 0, "TN" : 0, "FN" : 0}
        
        content = list(data)
        for each in content:
            label = each[1].strip().lower()
            assigned_label = "lib"

            if label == "lib" and assigned_label == "lib":
                measures["TP"] = measures["TP"] + 1
            elif label != "lib" and assigned_label == "lib":
                measures["FP"] = measures["FP"] + 1
            elif label != "lib" and assigned_label != "lib":
                measures["TN"] = measures["TN"] + 1
            elif label == "lib" and assigned_label != "lib":
                measures["FN"] = measures["FN"] + 1
                    
        #print(measures)
        precision = measures["TP"]/(measures["TP"] + measures["FP"])
        recall = measures["TP"]/(measures["TP"] + measures["FN"])
        f1 = 2 * (precision * recall)/(precision + recall)
        
        add_precision = add_precision + precision
        add_recall = add_recall + recall
        add_f1 = add_f1 + f1

        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f1))
        
        fol_counter = fol_counter + 1
        print("-------------------------------------")
        
    print("Overall:")
    print("Precision: ")
    print(add_precision/folds)
    print("Recall: ")
    print(add_recall/folds)
    print("F1: ")
    print(add_f1/folds)


def within_library():
    lib_names = ['boto', 'docker', 'flask', 'mock', 'pandas']

    base_folder = '../data/within/'
    all_precision, all_recall, all_f1 = 0, 0, 0

    for lib in lib_names:
        folders = os.listdir(base_folder + lib)
        fol_counter = 1

        add_precision = 0.0
        add_recall = 0.0
        add_f1 = 0.0

        folds = 5

        for fol in folders:
            fol_path = base_folder + lib + '/' + str(fol_counter) + "/test.json"

            print("Fold: " + str(fol_counter))
            with open(fol_path) as json_file:
                data = json.load(json_file)

            measures = {"TP" : 0, "FP" : 0, "TN" : 0, "FN" : 0}
            
            content = list(data)
            for each in content:
                label = each[1].strip().lower()
                assigned_label = "lib"
                
                if label == "lib" and assigned_label == "lib":
                    measures["TP"] = measures["TP"] + 1
                elif label != "lib" and assigned_label == "lib":
                    measures["FP"] = measures["FP"] + 1
                elif label != "lib" and assigned_label != "lib":
                    measures["TN"] = measures["TN"] + 1
                elif label == "lib" and assigned_label != "lib":
                    measures["FN"] = measures["FN"] + 1
                        
            # print(measures)
            precision = measures["TP"]/(measures["TP"] + measures["FP"])
            recall = measures["TP"]/(measures["TP"] + measures["FN"])
            f1 = 2 * (precision * recall)/(precision + recall)
            
            add_precision = add_precision + precision
            add_recall = add_recall + recall
            add_f1 = add_f1 + f1

            # print("Precision: " + str(precision))
            # print("Recall: " + str(recall))
            # print("F1: " + str(f1))
            
            fol_counter = fol_counter + 1
            print("-------------------------------------")
        
        all_precision += (add_precision / folds)
        all_recall += (add_recall / folds)
        all_f1 += (add_f1 / folds)

        print("Overall:")
        print("Precision: ")
        print(add_precision/folds)
        print("Recall: ")
        print(add_recall/folds)
        print("F1: ")
        print(add_f1/folds)
    print('precision: ', round(all_precision / 5, 2))
    print('recall: ', round(all_recall / 5, 2))
    print('f1: ', round(all_f1 / 5, 2))

def cross_library():
    base_folder = '../data/cross/'

    folders = os.listdir(base_folder)
    fol_counter = 1
    add_precision = 0.0
    add_recall = 0.0
    add_f1 = 0.0
    folds = 23

    for fol in folders:
        fol_path = base_folder + str(fol_counter) + "/test.json"

        print("Fold: " + str(fol_counter))
        with open(fol_path) as json_file:
            data = json.load(json_file)

        measures = {"TP" : 0, "FP" : 0, "TN" : 0, "FN" : 0}
        
        content = list(data)
        for each in content:
            label = each[1].strip().lower()
            assigned_label = "lib"

            if label == "lib" and assigned_label == "lib":
                measures["TP"] = measures["TP"] + 1
            elif label != "lib" and assigned_label == "lib":
                measures["FP"] = measures["FP"] + 1
            elif label != "lib" and assigned_label != "lib":
                measures["TN"] = measures["TN"] + 1
            elif label == "lib" and assigned_label != "lib":
                measures["FN"] = measures["FN"] + 1
                    
        #print(measures)
        precision = measures["TP"]/(measures["TP"] + measures["FP"])
        recall = measures["TP"]/(measures["TP"] + measures["FN"])
        f1 = 2 * (precision * recall)/(precision + recall)
        
        add_precision = add_precision + precision
        add_recall = add_recall + recall
        add_f1 = add_f1 + f1

        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f1))
        
        fol_counter = fol_counter + 1
        print("-------------------------------------")
        
    print("Overall:")
    print("Precision: ")
    print(add_precision/folds)
    print("Recall: ")
    print(add_recall/folds)
    print("F1: ")
    print(add_f1/folds)

if __name__ == '__main__':
    # cross_library()
    within_library()
    # mixed_library()