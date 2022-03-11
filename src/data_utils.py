from transformers import AutoTokenizer
from config import max_len
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, data, labels, model):
        self.data = data
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tweet = str(self.data[index])

        encoded = self.tokenizer(
            tweet,
            max_length=max_len,
            padding='max_length',  # Pad to max_length
            truncation=True,  # Truncate to max_length
            return_tensors='pt'
        )

        token_ids = encoded['input_ids'].squeeze(0)
        attn_masks = encoded['attention_mask'].squeeze(0)

        label = self.labels[index]
        return token_ids, attn_masks, label  