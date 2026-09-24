import json
from pathlib import Path

import pytest

from scripts.train_careguide_intent_transformer import LABELS, class_weights, read_jsonl


def test_transformer_training_contract_never_combines_splits(tmp_path: Path):
    rows = [{"split": "train", "label": label, "message": label} for label in LABELS]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")
    loaded = read_jsonl(path, "train")
    assert len(loaded) == len(LABELS)
    assert class_weights(loaded) == [1.0] * len(LABELS)
    assert class_weights(loaded, power=0.0) == [1.0] * len(LABELS)
    with pytest.raises(ValueError, match="mixed split"):
        read_jsonl(path, "validation")
