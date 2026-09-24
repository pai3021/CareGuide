"""Disk-backed lexical retrieval baseline for the 960k-document KUAKE-IR corpus."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from careguide.retrieval.bm25_index import tokenize_knowledge_text


def _fts_query(text: str) -> str:
    terms = list(dict.fromkeys(tokenize_knowledge_text(text)))
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


class KuakeIRFTSIndex:
    """SQLite FTS5 index that keeps benchmark data outside process memory."""

    version = "careguide-kuake-ir-sqlite-fts5-cjk-bigram-v1"

    def __init__(self, path: Path):
        self.path = Path(path)

    def build(self, corpus_path: Path, *, batch_size: int = 2_000) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".building")
        temporary.unlink(missing_ok=True)
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=MEMORY;
                CREATE TABLE documents(rowid INTEGER PRIMARY KEY, doc_id TEXT NOT NULL UNIQUE);
                CREATE VIRTUAL TABLE documents_fts USING fts5(
                    tokens, content='documents', content_rowid='rowid', tokenize='unicode61'
                );
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            documents: list[tuple[int, str]] = []
            postings: list[tuple[int, str]] = []
            count = 0
            with Path(corpus_path).open("r", encoding="utf-8") as handle:
                for count, line in enumerate(handle, 1):
                    doc_id, text = line.rstrip("\r\n").split("\t", 1)
                    documents.append((count, doc_id))
                    postings.append((count, " ".join(tokenize_knowledge_text(text))))
                    if len(documents) >= batch_size:
                        connection.executemany("INSERT INTO documents(rowid,doc_id) VALUES (?,?)", documents)
                        connection.executemany("INSERT INTO documents_fts(rowid,tokens) VALUES (?,?)", postings)
                        connection.commit()
                        documents.clear(); postings.clear()
            if documents:
                connection.executemany("INSERT INTO documents(rowid,doc_id) VALUES (?,?)", documents)
                connection.executemany("INSERT INTO documents_fts(rowid,tokens) VALUES (?,?)", postings)
            connection.execute("INSERT INTO metadata(key,value) VALUES ('version',?)", (self.version,))
            connection.execute("INSERT INTO metadata(key,value) VALUES ('document_count',?)", (str(count),))
            connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('optimize')")
            connection.commit()
        finally:
            connection.close()
        temporary.replace(self.path)
        return count

    def document_count(self) -> int:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key='document_count'").fetchone()
        return int(row[0]) if row else 0

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        expression = _fts_query(query)
        if not expression:
            return []
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """SELECT documents.doc_id, bm25(documents_fts) AS score
                   FROM documents_fts JOIN documents ON documents.rowid=documents_fts.rowid
                   WHERE documents_fts MATCH ? ORDER BY score LIMIT ?""",
                (expression, max(1, int(top_k))),
            ).fetchall()
        return [(str(doc_id), float(score)) for doc_id, score in rows]


def evaluate_queries(index: KuakeIRFTSIndex, queries: dict[str, str],
                     relevance: dict[str, str], *, top_k: int = 10) -> dict:
    rows = []
    for query_id, query in queries.items():
        returned = [doc_id for doc_id, _ in index.search(query, top_k=top_k)]
        gold = relevance[query_id]
        rank = returned.index(gold) + 1 if gold in returned else None
        rows.append({"query_id": query_id, "gold_doc_id": gold, "returned": returned,
                     "rank": rank})
    total = len(rows) or 1
    recall = lambda cutoff: sum(bool(row["rank"] and row["rank"] <= cutoff) for row in rows) / total
    return {
        "case_count": len(rows),
        "mrr_at_10": sum(1 / row["rank"] if row["rank"] else 0 for row in rows) / total,
        "recall_at_1": recall(1), "recall_at_5": recall(5), "recall_at_10": recall(10),
        "ndcg_at_10": sum(1 / math.log2(row["rank"] + 1) if row["rank"] else 0
                           for row in rows) / total,
        "rows": rows,
    }
