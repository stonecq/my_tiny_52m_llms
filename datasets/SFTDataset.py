import json
import pickle
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer


class SFTDataset(Dataset):
    def __init__(self, data_path_list: List, max_len:int = 256, tokenizer=None, memmap=False):
        super().__init__()
        self.max_len = max_len
        self.data = []

        raw_data = []
        for data_path in data_path_list:
            with open(data_path, 'rb') as f:
                raw_data.extend(pickle.load(f))
        self.data = [item for item in raw_data if len(item) <= max_len]

        del raw_data
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/llama/work_space/my_llms_learn/my_tiny_llms/tokenizer/chatglm", trust_remote_code=True)
        self.eos = self.tokenizer.special_tokens['<eos>']
        self.bos = self.tokenizer.special_tokens['<bos>']
        self.pad = self.tokenizer.special_tokens['<pad>']  if self.tokenizer.special_tokens['<pad>'] else 0


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = self.data[index]
        question_len = input_ids.index(self.bos)
        pad_len = self.max_len - len(input_ids)
        attention_mask = len(input_ids) * [1] + [0] * pad_len
        loss_mask = (question_len + 1) * [0] + [1] * (len(input_ids[question_len+1:])) + [0] * pad_len

        input_ids = np.array(input_ids + pad_len * [self.pad])
        X = np.array(input_ids[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:])
        attention_mask = np.array(attention_mask[:-1])
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask), torch.from_numpy(attention_mask)