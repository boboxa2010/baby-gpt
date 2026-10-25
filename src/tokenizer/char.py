import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from .base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    unk_token_id = 0

    def __init__(self, alphabet: Sequence[str]):
        self.alphabet = sorted(set(alphabet))
        self.char_to_id = {char: i + 1 for i, char in enumerate(self.alphabet)}
        self.id_to_char = dict(enumerate(["<unk>", *self.alphabet]))

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(text)

    @classmethod
    def from_iterator(cls, texts: Iterable[str]) -> "CharTokenizer":
        characters = set()
        for text in texts:
            characters.update(text)
        return cls(list(characters))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(char, self.unk_token_id) for char in text]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        return "".join(
            self.id_to_char[token_id]
            for token_id in map(int, ids)
            if not (skip_special_tokens and token_id == self.unk_token_id)
        )

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "char_tokenizer.json").write_text(
            json.dumps({"alphabet": self.alphabet}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "CharTokenizer":
        config = json.loads(
            (Path(path) / "char_tokenizer.json").read_text(encoding="utf-8")
        )
        return cls(config["alphabet"])
