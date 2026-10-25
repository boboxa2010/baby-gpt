from collections.abc import Iterable
from pathlib import Path

from .base_tokenizer import BaseTokenizer


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, pretrained_model_name_or_path: str | Path, **kwargs):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(pretrained_model_name_or_path), **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def unk_token_id(self) -> int | None:
        return self.tokenizer.unk_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text, add_special_tokens=False, padding=False, truncation=False
        )

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(
            list(ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def save_pretrained(self, directory: str | Path) -> None:
        self.tokenizer.save_pretrained(str(directory))

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> "HuggingFaceTokenizer":
        return cls(path, **kwargs)
