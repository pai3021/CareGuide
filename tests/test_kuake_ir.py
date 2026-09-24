from pathlib import Path

from careguide.evaluation.kuake_ir import KuakeIRFTSIndex, evaluate_queries


def test_disk_backed_kuake_ir_index_and_metrics(tmp_path: Path):
    corpus = tmp_path / "corpus.tsv"
    corpus.write_text("d1\t持续胸痛需要尽快就医\nd2\t感冒常见症状包括鼻塞\nd3\t胸痛挂什么科\n", encoding="utf-8")
    index = KuakeIRFTSIndex(tmp_path / "index.sqlite3")
    assert index.build(corpus, batch_size=2) == 3
    assert index.document_count() == 3
    hits = index.search("胸痛就医", top_k=2)
    assert hits and hits[0][0] in {"d1", "d3"}
    report = evaluate_queries(index, {"q1": "感冒鼻塞"}, {"q1": "d2"})
    assert report["case_count"] == 1
    assert report["recall_at_1"] == 1.0
    assert report["mrr_at_10"] == 1.0
