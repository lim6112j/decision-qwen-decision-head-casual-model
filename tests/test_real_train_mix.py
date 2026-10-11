"""Real samples round-trip and curriculum mixing (weight + opt-in path)."""

from types import SimpleNamespace

import numpy as np
import torch

from decision_lab.cli import _load_real_samples
from decision_lab.head.dynamic_train import DynamicTrainingSample
from decision_lab.real.build_dataset import load_samples

DIM = 8


def _sample(gold_idx=0):
    return DynamicTrainingSample(
        state_fields_emb=np.zeros((2, DIM), dtype=np.float32),
        option_embs=np.zeros((3, DIM), dtype=np.float32),
        gold_idx=gold_idx, question_type="choice", question_text="q",
        question_emb=np.zeros(DIM, dtype=np.float32), is_variant=True,
    )


def test_samples_round_trip(tmp_path):
    path = tmp_path / "samples.pt"
    torch.save([_sample(1), _sample(2)], path)
    loaded = load_samples(path)
    assert [s.gold_idx for s in loaded] == [1, 2]
    assert all(s.is_variant for s in loaded)


def test_empty_path_is_opt_out(tmp_path):
    dcfg = SimpleNamespace(real_data_path="", real_weight=3)
    assert _load_real_samples(dcfg, tmp_path) == []


def test_missing_file_warns_and_skips(tmp_path):
    dcfg = SimpleNamespace(real_data_path=str(tmp_path / "nope.pt"), real_weight=1)
    assert _load_real_samples(dcfg, tmp_path) == []


def test_weight_oversamples(tmp_path):
    path = tmp_path / "samples.pt"
    torch.save([_sample(0)], path)
    dcfg = SimpleNamespace(real_data_path=str(path), real_weight=3)
    assert len(_load_real_samples(dcfg, tmp_path)) == 3