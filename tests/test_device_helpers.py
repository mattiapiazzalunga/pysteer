from __future__ import annotations

import torch
import torch.nn as nn

from steering_engine.executor_services import ExecutorTokenizerIO


def test_as_torch_device_cuda_string_falls_back_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert ExecutorTokenizerIO.as_torch_device("cuda:1").type == "cpu"
    assert ExecutorTokenizerIO.as_torch_device(torch.device("cuda:0")).type == "cpu"


def test_infer_input_device_uses_hf_device_map_for_meta_embeddings(monkeypatch):
    class MetaEmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4, device="meta")
            self.hf_device_map = {
                "model.embed_tokens": "cuda:1",
                "model.layers.0": "cuda:0",
            }

        def get_input_embeddings(self):
            return self.embed

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert ExecutorTokenizerIO.infer_input_device(MetaEmbeddingModel()) == torch.device("cuda:1")


def test_infer_input_device_never_returns_meta_without_device_map():
    class MetaOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, device="meta"))

    assert ExecutorTokenizerIO.infer_input_device(MetaOnlyModel()).type == "cpu"
