# Structure

This is a modified implementation of an existing approach called APIReal from- https://ai.tencent.com/ailab/media/publications/empirical/APIReal_An_API_Recognition_and_Linking_Approach_for.pdf

Refer to `APIReal_README.md` for the pre-requisites and setup.

## Preliminaries:

We provide `jsontoconll.py`, a modified version of `texttoconll.py` from the original code.

```
python jsontoconll.py
```

## Usage:

For each training and test file, follow the following by changing the setting/folder and number accordingly.

```bash
python enner.py bc-ce < api_recog/stratified/training1.conll > api_recog/stratified/training1.data

python enner.py bc-ce < api_recog/stratified/test1.conll > api_recog/stratified/test1.data

./crfsuite-0.12/bin/crfsuite learn -m ./models/stratified/model_1 api_recog/stratified/training1.data

./crfsuite-0.12/bin/crfsuite tag -m ./models/stratified/model_1 api_recog/stratified/test1.data > api_recog/stratified/res/output1.data
```

## Evaluating the predictions:

Repeat the run for each test file.

```
python json_tweet_mapping.py
```
