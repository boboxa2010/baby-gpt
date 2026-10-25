import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from src.tokenizer import CharTokenizer, HuggingFaceTokenizer

from .base_dataset import BaseDataset


class ShakespeareDataset(BaseDataset):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    def __init__(
        self,
        data_dir,
        split,
        block_size,
        tokenizer_name="char",
        train_fraction=0.9,
        revision=None,
        stride=1,
        limit=None,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer_name = str(tokenizer_name)
        self.train_fraction = train_fraction
        self.revision = revision
        self.prepare()
        tokens_path = {
            "train": self.cache_dir / "train.bin",
            "val": self.cache_dir / "val.bin",
        }[split]
        super().__init__(tokens_path, block_size, stride=stride, limit=limit)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def prepare(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.data_dir / "input.txt"
        if not source_path.exists():
            urlretrieve(self.url, source_path)

        text = source_path.read_bytes().decode("utf-8")
        settings = {
            "tokenizer_name": self.tokenizer_name,
            "revision": self.revision,
            "train_fraction": self.train_fraction,
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        cache_key = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()[:16]
        self.cache_dir = self.data_dir / "processed" / cache_key
        tokenizer_dir = self.cache_dir / "tokenizer"
        metadata_path = self.cache_dir / "metadata.json"
        tokenizer_class = (
            CharTokenizer if self.tokenizer_name == "char" else HuggingFaceTokenizer
        )

        if all(
            (self.cache_dir / name).exists()
            for name in ("train.bin", "val.bin", "metadata.json")
        ):
            self.tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return

        boundary = int(len(text) * self.train_fraction)
        splits = {"train": text[:boundary], "val": text[boundary:]}
        self.tokenizer = (
            CharTokenizer.from_text(splits["train"])
            if self.tokenizer_name == "char"
            else HuggingFaceTokenizer(self.tokenizer_name, revision=self.revision)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(tokenizer_dir)
        counts = {}
        for split, split_text in splits.items():
            tokens = np.asarray(self.tokenizer.encode(split_text), dtype=np.uint32)
            tokens.tofile(self.cache_dir / f"{split}.bin")
            counts[split] = len(tokens)

        self.metadata = {
            **settings,
            "dtype": "uint32",
            "vocab_size": self.vocab_size,
            "num_tokens": counts,
        }
        metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
