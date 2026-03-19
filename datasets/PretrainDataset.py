from typing import List

import numpy as np
import torch
from transformers.pipelines.question_answering import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path_list: List, max_length: int=256, memmap=False):
        super().__init__()
        if memmap: # 内存映射
            with open(data_path_list[0], 'r') as f:
                nbytes = f.seek(0, 2)
                flen = f.tell() // np.dtype('uint16').itemsize
            self.data = np.memmap(data_path_list[0], dtype=np.dtype('uint16'), shape=(flen // max_length, max_length))
        else:
            data_list = []
            for data_path in data_path_list:
                with open(data_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)
                    data_list.append(data)

            data = np.concatenate(data_list)
            data = data[:max_length * int(len(data) / max_length)]
            self.data = data.reshape(-1, max_length)
        self.total_tokens = self.data.size
        print("memmap:{} train data.shape:{}".format(memmap, self.data.shape))
        print(f"\n" + "=" * 40)
        print(f"数据加载成功!")
        print(f"模式: {'memmap' if memmap else 'Memory'}")
        print(f"总 Token 数: {self.total_tokens:,}")
        print(f"总 Token 数 (M): {self.total_tokens / 1e6:.2f} M")
        print(f"总训练行数 (Samples): {self.data.shape[0]:,}")
        print("=" * 40 + "\n")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(Y)