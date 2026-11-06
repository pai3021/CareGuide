"""Sparse output semantics; relevance is measured separately on actual dev data."""
import pytest
import json
import sqlite3

from careguide.corpus import Corpus
from careguide.retrieval.bge_sparse import token_weights_to_sparse, LocalBGESparseRetriever


def test_bge_sparse_excludes_specials_and_keeps_maximum_repeated_token_weight():
    assert token_weights_to_sparse([0, 9, 9, 12, 2, 1, 3],
                                   [100, .2, .8, -.1, 90, 80, 70], {0, 1, 2, 3}) == {'9': .8}


@pytest.mark.parametrize('ids,weights', [([9], []), ([9], [float('nan')]), ([9], [float('inf')])])
def test_invalid_model_output_is_not_silently_cached(ids, weights):
    with pytest.raises(ValueError):
        token_weights_to_sparse(ids, weights, set())


@pytest.mark.parametrize('fingerprint', ['wrong', None])
def test_runtime_rejects_foreign_or_incomplete_document_checkpoint_before_model_load(tmp_path, fingerprint):
    corpus = Corpus()
    model = tmp_path/'model'
    model.mkdir()
    (model/'careguide_model_lock.json').write_text(json.dumps({'revision': 'test', 'files': {}}))
    checkpoint = tmp_path/'vectors.sqlite3'
    with sqlite3.connect(checkpoint) as db:
        db.execute('create table config (identity text)')
        db.execute('create table vectors (id text, text_sha256 text, weights text)')
        db.execute('insert into config values (?)', (json.dumps({
            'corpus_fingerprint': fingerprint or corpus.fingerprint,
            'model': 'BAAI/bge-m3', 'revision': 'test', 'files': {}}),))
    with pytest.raises(ValueError, match='identity mismatch'):
        LocalBGESparseRetriever(corpus, model, checkpoint)
