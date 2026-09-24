import numpy as np

from careguide.external_data import QIC_LABELS
from scripts.run_qic_remote_embedding_evaluation import (confusion, cosine_matrix, fingerprint,
                                                          normalize,
                                                          predictions_from_scores,
                                                          row_standardize)


def test_qic_embedding_helpers_are_deterministic():
    rows = ({"query": "头痛是什么原因", "label": "病因分析"},)
    assert fingerprint(rows) == fingerprint(rows)
    vectors = normalize(np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(
        cosine_matrix(vectors, vectors),
        np.asarray([[1.0, 0.8], [0.8, 1.0]], dtype=np.float32),
    )
    assert predictions_from_scores(np.eye(2, len(QIC_LABELS)))[:2] == list(QIC_LABELS[:2])
    standardized = row_standardize(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert np.allclose(standardized.mean(axis=1), 0.0)
    assert np.allclose(standardized.std(axis=1), 1.0)


def test_confusion_uses_gold_rows_and_predicted_columns():
    matrix = confusion(["病因分析", "病因分析"], ["病因分析", "疾病表述"])
    assert matrix["病因分析"]["病因分析"] == 1
    assert matrix["病因分析"]["疾病表述"] == 1
