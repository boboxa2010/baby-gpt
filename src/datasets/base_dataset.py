from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, tokens_path, block_size, dtype="uint32", stride=1, limit=None):
        tokens_path = Path(tokens_path)
        self.tokens = (
            np.memmap(tokens_path, dtype=dtype, mode="r")
            if tokens_path.stat().st_size
            else np.empty(0, dtype=dtype)
        )
        self.block_size = block_size
        index = range(0, max(0, len(self.tokens) - block_size), stride)
        self._index = index if limit is None else index[:limit]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, ind):
        window = self.load_window(self._index[ind])
        return {"input_ids": window[:-1], "labels": window[1:]}

    def load_window(self, start):
        window = self.tokens[start : start + self.block_size + 1]
        return torch.from_numpy(window.astype(np.int64))
