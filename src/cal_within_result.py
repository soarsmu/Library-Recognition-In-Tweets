# BERT

import re

relative_folder = '../'
bert1 = 'log/within_boto_BERT_2022-01-19-16:05:05.log'
bert2 = 'log/within_docker_BERT_2022-01-19-16:10:31.log'
bert3 = 'log/within_flask_BERT_2022-01-19-16:18:38.log'
bert4 = 'log/within_mock_BERT_2022-01-19-16:28:17.log'
bert5 = 'log/within_pandas_BERT_2022-01-19-16:38:35.log'

o1 = 'log/within_boto_BERTOverflow_2022-01-19-16:06:21.log'
o2 = 'log/within_docker_BERTOverflow_2022-01-19-16:12:28.log'
o3 = 'log/within_flask_BERTOverflow_2022-01-19-16:20:26.log'
o4 = 'log/within_mock_BERTOverflow_2022-01-19-16:30:14.log'
o5 = 'log/within_pandas_BERTOverflow_2022-01-19-16:40:27.log'

t1 = 'log/within_boto_BERTweet_2022-01-19-16:07:43.log'
t2 = 'log/within_docker_BERTweet_2022-01-19-16:14:33.log'
t3 = 'log/within_flask_BERTweet_2022-01-19-16:22:21.log'
t4 = 'log/within_mock_BERTweet_2022-01-19-16:32:16.log'
t5 = 'log/within_pandas_BERTweet_2022-01-19-16:42:27.log'

roberta1 = 'log/within_boto_RoBERTa_2022-01-19-16:09:04.log'
roberta2 = 'log/within_docker_RoBERTa_2022-01-19-16:16:32.log'
roberta3 = 'log/within_flask_RoBERTa_2022-01-19-16:24:14.log'
roberta4 = 'log/within_mock_RoBERTa_2022-01-19-16:34:16.log'
roberta5 = 'log/within_pandas_RoBERTa_2022-01-19-16:44:23.log'

x1 = 'log/within_boto_xlnet_2022-01-19-19:02:21.log'
x2 = 'log/within_docker_xlnet_2022-01-19-19:04:06.log'
x3 = 'log/within_flask_xlnet_2022-01-19-16:26:05.log'
x4 = 'log/within_mock_xlnet_2022-01-19-16:36:13.log'
x5 = 'log/within_pandas_xlnet_2022-01-19-16:46:15.log'

def average(file_list):
    precision, recall, f1 = 0, 0, 0
    for file in file_list:
        print(file)
        with open(relative_folder + file) as f:
            lines = f.readlines()
        pl = re.findall('precision: \[.+\]', lines[-3])
        precision += float(re.findall("\d+\.\d+", pl[0])[0])
        cur_p = round(float(re.findall("\d+\.\d+", pl[0])[0]), 2)
        # print('precision: {}'.format(cur_p))

        rl = re.findall('recall: \[.+\]', lines[-2])
        recall += float(re.findall("\d+\.\d+", rl[0])[0])
        cur_r = round(float(re.findall("\d+\.\d+", rl[0])[0]), 2)
        # print('recall: {}'.format(cur_r))

        fl = re.findall('f1_score: \[.+\]', lines[-1])
        f1 += float(re.findall("\d+\.\d+", fl[0])[0])
        cur_f = round(float(re.findall("\d+\.\d+", fl[0])[0]), 2)
        print('f1: {}'.format(cur_f))

    # print("precision: ", round(precision / 5, 2))
    # print("recall: ", round(recall / 5, 2))
    # print("f1_score: ", round(f1 / 5, 2))

if __name__ == '__main__':
    print('===========>BERT<===========')
    average([bert1, bert2, bert3, bert4, bert5])
    print('===========>RoBERTa<==========')
    average([roberta1, roberta2, roberta3, roberta4, roberta5])
    print('===========>BERTweet<===========')
    average([t1, t2, t3, t4, t5])
    print('============>BERTOverflow<=========')
    average([o1, o2, o3, o4, o5])
    print('============>xlnet<=============')
    average([x1, x2, x3, x4, x5])