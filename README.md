# Structure
# Data
By following Twitter data sharing recommendation and policy, check here https://developer.twitter.com/en/developer-terms/more-on-restricted-use-cases, we can only share the tweet ids.

Please check the corresponding tweet by using Twitter API. One simple way is to append the tweet_id after `https://twitter.com/i/web/status/`, you will be able to see the corresponding tweet.

Our data is under `./public_data`.

Take `./public_data/absl.txt` as an example: as the following image shows, the first column is the tweet id, and the second column is the label: 'Text' indicates it is not library-related; 'Lib' indicates it is a library-related.

![image](./public_data/example.png) 


## Pre-trained Transformer Models
0. Install the required packages
```
$ pip install -r requirements.txt
```
Go to the `./src` folder

1. Preprocess the data first
```
$ python preprocessing.py
```

2. within-library setting
```
python BERT_within_library.py --variant BERT --lib_name boto --seed 42
```

3. cross-library setting
```
python BERT_cross.py --variant BERT --seed 42
```

4. mixed setting
```
python BERT_stratified.py --variant BERT --seed 42
```

For mixed- and cross- settings, the performance will be directly logged inside the log files. For within-library setting, in order to get the performance across five libraries, please run the script `./src/cal_within_result.py`.

Note: we stated we used Sigmoid activation, and we implemented it with Softmax. In the binary classification setting, softmax is identical to sigmoid. [Reference](https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/#convergence)

## Baselines
### Strawman
The script is also inside `./src`.

```
python strawman.py
```

### Prasetyo et al.
The script is also inside `./src`.

```
python Prasetyo-approach --variant cross
```

### APIReal
Please refer to `./APIReal/README.md` for more details. You can find the original repo [here](https://github.com/baolingfeng/APIExing).

After getting the predictions, please run the script inside `./src` to calculate precision, recall and F1.

```
python cal_apireal_result.py
```

# Acknowledgement
We appreciate APIReal authors made their replication package public.

# Citation
If you find this repo useful, please consider to cite our work.

```
@inproceedings{zhang2022benchmarking,
  author = {Zhang, Ting and Chandrasekaran, Divya Prabha and Thung, Ferdian and Lo, David},
  title = {Benchmarking Library Recognition in Tweets},
  year = {2022},
  isbn = {9781450392983},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3524610.3527916},
  doi = {10.1145/3524610.3527916},
  abstract = {Software developers often use social media (such as Twitter) to share programming knowledge such as new tools, sample code snippets, and tips on programming. One of the topics they talk about is the software library. The tweets may contain useful information about a library. A good understanding of this information, e.g., on the developer's views regarding a library can be beneficial to weigh the pros and cons of using the library as well as the general sentiments towards the library. However, it is not trivial to recognize whether a word actually refers to a library or other meanings. For example, a tweet mentioning the word "pandas" may refer to the Python pandas library or to the animal. In this work, we created the first benchmark dataset and investigated the task to distinguish whether a tweet refers to a programming library or something else. Recently, the pre-trained Transformer models (PTMs) have achieved great success in the fields of natural language processing and computer vision. Therefore, we extensively evaluated a broad set of modern PTMs, including both general-purpose and domain-specific ones, to solve this programming library recognition task in tweets. Experimental results show that the use of PTM can outperform the best-performing baseline methods by 5% - 12% in terms of F1-score under within-, cross-, and mixed-library settings.},
  booktitle = {Proceedings of the 30th IEEE/ACM International Conference on Program Comprehension},
  pages = {343–353},
  numpages = {11},
  keywords = {disambiguation, software libraries, tweets, benchmark study},
  location = {Virtual Event},
  series = {ICPC '22}
}
```
