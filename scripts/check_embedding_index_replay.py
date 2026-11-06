"""Check native scoped search against actual cached 4B experiment responses.

No model client, configuration file, or external request. The saved provider
response is replayed only after native candidate IDs exactly match its inputs.
"""
import argparse
import json
import sqlite3
from contextlib import closing
from datetime import date

import numpy as np

from careguide.corpus import Corpus
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, versioned_collection_name
from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json
from scripts.compare_rag_embeddings import DIMENSIONS, MODEL, experiment_dir
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases


def main():
    from pathlib import Path
    import chromadb
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=read_json(args.plan)
    work=experiment_dir(args.work_dir)
    responses=read_json(work/'rerank-responses.json')['completed']
    saved={row['case_id']:row for row in responses}
    cases,_=load_cases('dev')
    chosen=[next(case for case in cases if case['id']==key) for key in plan['case_ids'][:3]]
    with closing(sqlite3.connect((work/'vectors.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        vectors={case['query']:np.frombuffer(db.execute('select vector from vectors where kind=? and id=?',
                 ('query',case['id'])).fetchone()[0],dtype='<f4').tolist() for case in chosen}
    captured=[]
    class CachedEmbedding:
        model=MODEL
        dimensions=DIMENSIONS
        def embed_query(self,query):return vectors[query]
    class CachedReranker:
        model='Qwen/Qwen3-Reranker-4B'
        def rerank(self,query,candidates,*,top_n):
            case=next(case for case in chosen if case['query']==query)
            row=saved[case['id']]
            ids=[key for key,_ in candidates]
            if ids!=row['candidate_ids']:
                raise ValueError('Native candidate inputs differ from the actual cached model request')
            captured.append(case['id'])
            return row['ids'][:top_n],row['scores'][:top_n],{}
    corpus=Corpus(CORPUS_DIR)
    client=chromadb.PersistentClient(path=str(work/'chroma'),settings=chromadb.Settings(anonymized_telemetry=False))
    knowledge=MedicalKnowledgeBase(corpus,mode='hybrid',embedding=CachedEmbedding(),
        collection=client.get_collection(versioned_collection_name(corpus),embedding_function=None),
        reranker=CachedReranker(),rerank_candidates=40,hybrid_candidates=100)
    outcomes=[]
    for case in chosen:
        hits=knowledge.search(case['query'],list(corpus.records),today=date(2026,11,6),top_k=10)
        assert [hit['record_id'] for hit in hits]==saved[case['id']]['ids'][:10]
        outcomes.append({'id':case['id'],'candidate_inputs_match':True,'final_ids_match':True})
    assert len(captured)==3
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,{'external_requests':0,'model':MODEL,'dimensions':DIMENSIONS,
                            'outcomes':outcomes,'scope':'Three native candidate/input replay checks, not additional quality evaluation.'})
    print(json.dumps({'native_replay_checks_passed':len(outcomes),'external_requests':0}))


if __name__=='__main__':main()
