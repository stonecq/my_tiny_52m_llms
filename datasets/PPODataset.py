import pickle
from typing import List

import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PPODataset(Dataset):
    def __init__(self, datasets_path_list:List[str], max_len:int=256, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.data = []
        raw_data = []
        for datasets_path in datasets_path_list:
            with open(datasets_path, 'rb') as f:
                 raw_data.extend(pickle.load(f))

        self.data = [item for item in raw_data if len(item)<max_len]
        if tokenizer==None:
            tokenizer = AutoTokenizer.from_pretrained(
                "/llama/work_space/my_llms_learn/my_tiny_llms/tokenizer/chatglm", trust_remote_code=True)
        self.bos = tokenizer.special_tokens["<bos>"]
        self.eos = tokenizer.special_tokens['<eos>']
        self.pad = tokenizer.special_tokens['<pad>'] if tokenizer.special_tokens['<pad>'] else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_input_ids = self.data[index] + [self.bos]
        # ppo 通常是左填充
        pad_len = self.max_len - len(raw_input_ids)
        attention_mask = [0] * pad_len + [1] * len(raw_input_ids)
        inputs_ids = [self.pad] * pad_len + raw_input_ids

        attention_mask = torch.from_numpy(np.array(attention_mask, dtype=np.int64))
        inputs_ids = torch.from_numpy(np.array(inputs_ids, dtype=np.int64))
        raw_input_ids = torch.tensor(raw_input_ids, dtype=torch.long)
        return inputs_ids, attention_mask, raw_input_ids
