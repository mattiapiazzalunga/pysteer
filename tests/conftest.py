from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


class TinyTokenizer:
    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = int(vocab_size)
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self._pad_token = "<pad>"
        self.pad_token_id = 0
        self.padding_side = "left"
        self.truncation_side = "right"
        self.model_max_length = 64

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str | None) -> None:
        self._pad_token = value
        if value is None:
            self.pad_token_id = None
        elif value == self.eos_token:
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = 0

    def _encode_one(self, text: str) -> list[int]:
        ids = [3 + (ord(ch) % (self.vocab_size - 3)) for ch in str(text) if not ch.isspace()]
        return ids or [3]

    def _truncate(self, ids: list[int], max_length: int | None) -> list[int]:
        if max_length is None or len(ids) <= max_length:
            return ids
        if self.truncation_side == "left":
            return ids[-max_length:]
        return ids[:max_length]

    def _pad_batch(self, rows: list[list[int]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row) for row in rows)
        out_ids = []
        out_mask = []
        for row in rows:
            pad = [self.pad_token_id] * (max_len - len(row))
            mask_pad = [0] * (max_len - len(row))
            if self.padding_side == "left":
                out_ids.append(pad + row)
                out_mask.append(mask_pad + [1] * len(row))
            else:
                out_ids.append(row + pad)
                out_mask.append([1] * len(row) + mask_pad)
        return {
            "input_ids": torch.tensor(out_ids, dtype=torch.long),
            "attention_mask": torch.tensor(out_mask, dtype=torch.long),
        }

    def __call__(
        self,
        texts,
        *,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        single = isinstance(texts, str)
        raw_texts = [texts] if single else list(texts)
        rows = [self._encode_one(t) for t in raw_texts]
        if truncation:
            rows = [self._truncate(row, max_length) for row in rows]

        if return_tensors == "pt":
            return self._pad_batch(rows)

        if single:
            return {"input_ids": rows[0], "attention_mask": [1] * len(rows[0])}
        return {
            "input_ids": rows,
            "attention_mask": [[1] * len(row) for row in rows],
        }

    def pad(self, items, *, padding=True, return_tensors=None):
        rows = [list(item["input_ids"]) for item in items]
        masks = [list(item.get("attention_mask", [1] * len(row))) for item, row in zip(items, rows)]
        max_len = max(len(row) for row in rows)
        out_ids = []
        out_masks = []
        for row, mask in zip(rows, masks):
            pad = [self.pad_token_id] * (max_len - len(row))
            mask_pad = [0] * (max_len - len(row))
            if self.padding_side == "left":
                out_ids.append(pad + row)
                out_masks.append(mask_pad + mask)
            else:
                out_ids.append(row + pad)
                out_masks.append(mask + mask_pad)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(out_ids, dtype=torch.long),
                "attention_mask": torch.tensor(out_masks, dtype=torch.long),
            }
        return {"input_ids": out_ids, "attention_mask": out_masks}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.eos_token if int(token_id) == self.eos_token_id else f"<tok{int(token_id)}>"


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int, scale: float) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((hidden_size,), float(scale)))

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states + self.bias.view(1, 1, -1)


class TinyCausalModel(nn.Module):
    def __init__(self, *, vocab_size: int = 32, hidden_size: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            pad_token_id=0,
            eos_token_id=2,
            max_position_embeddings=64,
            model_type="tiny",
        )
        self.generation_config = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(TinyBlock(hidden_size, i + 1) for i in range(num_layers))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, use_cache=False, labels=None, **kwargs):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        return SimpleNamespace(last_hidden_state=hidden, logits=logits)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
        bsz = int(input_ids.size(0))
        extra = torch.full(
            (bsz, int(max_new_tokens)),
            int(self.config.eos_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, extra], dim=1)


@pytest.fixture()
def tiny_tokenizer() -> TinyTokenizer:
    return TinyTokenizer()


@pytest.fixture()
def tiny_model() -> TinyCausalModel:
    torch.manual_seed(7)
    return TinyCausalModel()


@pytest.fixture()
def activation_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    acts = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    starts = torch.tensor([1, 1], dtype=torch.long)
    attention = torch.ones((2, 2), dtype=torch.bool)
    return acts, labels, starts, attention
