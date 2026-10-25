from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path


class BaseTokenizer(ABC):
    unk_token_id: int | None = None
    eos_token_id: int | None = None

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    def __len__(self) -> int:
        return self.vocab_size

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str: ...

    @abstractmethod
    def save_pretrained(self, directory: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str | Path) -> "BaseTokenizer": ...
