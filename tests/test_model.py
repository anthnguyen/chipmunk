from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from chipmunk.model import Runner


class _AddOne(nn.Module):
    def forward(self, hidden):
        return hidden + 1


class _OOMBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Identity()
        self.layers = nn.ModuleList([_AddOne()])

    def forward(self, input_ids, attention_mask):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        hidden = self.embed_tokens(hidden)
        # Fail after the embedding hook has fired to exercise partial-buffer
        # cleanup as well as recursive batch splitting.
        if input_ids.shape[0] > 2:
            raise torch.OutOfMemoryError("synthetic capture OOM")
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _OOMBase()


def test_capture_splits_oom_batches_without_reordering_or_partial_rows():
    runner = Runner.__new__(Runner)
    runner.device = "cpu"
    runner.model = _FakeCausalLM()
    runner.chat_ids = lambda _system, user: [int(user)]
    runner._pad_left = lambda seqs: (
        torch.tensor(seqs, dtype=torch.long),
        torch.ones((len(seqs), 1), dtype=torch.long),
    )

    prompts = [("system", str(i)) for i in range(5)]
    acts = runner.capture(prompts, [0, 1], batch_size=4)

    assert acts[0].shape == (5, 3)
    assert acts[1].shape == (5, 3)
    np.testing.assert_array_equal(acts[0][:, 0], np.arange(5))
    np.testing.assert_array_equal(acts[1][:, 0], np.arange(5) + 1)
