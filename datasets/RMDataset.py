import pickle
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer


class RMDatasets(Dataset):
    def __init__(self, data_path_list, max_length: int=256, tokenizer=None):
        super().__init__()
        self.max_len = max_length
        self.data = []

        raw_data = []
        for data_path in data_path_list:
            with open(data_path, 'rb') as f:
                raw_data.extend(pickle.load(f))

        self.data = [item for item in raw_data if len(item[0]) <= max_length and len(item[1]) <= max_length]
        del raw_data
        if tokenizer==None:
            tokenizer = AutoTokenizer.from_pretrained(
                "/llama/work_space/my_llms_learn/my_tiny_llms/tokenizer/chatglm", trust_remote_code=True)
        self.bos = tokenizer.special_tokens["<bos>"]
        self.eos = tokenizer.special_tokens['<eos>']
        self.pad = tokenizer.special_tokens['<pad>'] if tokenizer.special_tokens['<pad>'] else 0

    def _process(self, input_ids:List):
        pad_len = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = np.array(input_ids + [self.pad] * pad_len)
        X = np.array(input_ids[:-1]).astype(np.int64)
        attention_mask = np.array(attention_mask[:-1])
        return torch.from_numpy(X), torch.from_numpy(attention_mask)

    def __getitem__(self, index):
        input_ids = self.data[index]
        x_accept, attention_mask_accept = self._process(input_ids[0])
        x_reject, attention_mask_reject = self._process(input_ids[1])
        return x_accept, attention_mask_accept, x_reject, attention_mask_reject

    def __len__(self):
        return len(self.data)