"""Separate-index Qwen 0.6B/4B embedding ablation with fixed 4B reranking.

Prepare is entirely offline. Remote run requires explicit per-endpoint budgets.
Old indexes are never opened writable. All new vectors/checkpoints/index files
live under a fresh experiment directory; inference attempts are persisted first.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from careguide.retrieval.embedding_provider import OpenAICompatibleEmbeddingFunction
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, versioned_collection_name
from careguide.retrieval.reranker import OpenAICompatibleReranker
from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json, query_fingerprint
from scripts.compare_rag_rerank_models import ENGLISH_INSTRUCTION
from scripts.evaluate_rag_candidate_pools import BASELINE, partition_rows, rank
from scripts.evaluate_rag_score_fusion import validate_cache
from scripts.private_config import load_config
from scripts.rerank_rag_candidate_plan import AttemptLedger, paired_interval, sha
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, ROOT, load_cases
from scripts.snapshot_rag_index import tree_hashes

MODEL = 'Qwen/Qwen3-Embedding-4B'
DIMENSIONS = 1024
BATCH = 32


def inputs():
    corpus = Corpus(CORPUS_DIR)
    cases,manifest = load_cases('dev')
    docs = [(record.id,MedicalKnowledgeBase.retrieval_text(record)) for record in corpus.records]
    return corpus,cases,manifest,docs


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prepare(args):
    if args.plan.exists():
        raise ValueError('Plan already exists; use a new path')
    corpus,cases,manifest,docs = inputs()
    cache = read_json(args.baseline_scores)
    rows,names = validate_cache(cache)
    if (set(names) != {'dense','bm25'} or cache['metadata']['embedding_model'] != 'Qwen/Qwen3-Embedding-0.6B'
            or cache['metadata']['embedding_dimensions'] != DIMENSIONS):
        raise ValueError('Expected genuine 0.6B/1024 baseline scores')
    old_plan,old_responses = read_json(args.baseline_plan),read_json(args.baseline_responses)
    if old_responses['identity']['plan_sha256'] != sha(args.baseline_plan):
        raise ValueError('Old reranker responses do not match their plan')
    baseline = [r for r in old_responses['completed'] if r['arm']=='english_4b40']
    expected_arm={'model':'Qwen/Qwen3-Reranker-4B','instruction':ENGLISH_INSTRUCTION,'pool_size':40,'cached':False}
    if old_plan['arms']['english_4b40']!=expected_arm:
        raise ValueError('Baseline must use the same 4B reranker/instruction/pool')
    if len(baseline) != 100 or {r['case_id'] for r in baseline} != set(old_plan['case_ids']):
        raise ValueError('Expected complete fixed 100-query 4B reranker baseline')
    by_id = {row['id']:row for row in rows}
    for row in baseline:
        if row['candidate_ids'] != rank(by_id[row['case_id']],BASELINE)[:40]:
            raise ValueError('Old reranker pool differs from the baseline score cache')
    snapshot = read_json(args.baseline_snapshot)
    if (not snapshot['source_unchanged'] or snapshot['verified_collection']['record_count'] != len(docs)
            or snapshot['verified_collection']['metadata']['corpus_fingerprint'] != corpus.fingerprint):
        raise ValueError('Certified 0.6B index snapshot is required')
    if tree_hashes(Path(snapshot['destination']))!=snapshot['files']:
        raise ValueError('Preserved 0.6B index snapshot files changed')
    document_fingerprint = text_hash(json.dumps([(key,text_hash(text)) for key,text in docs]))
    plan = {
        'version':'careguide-embedding-ablation-v1','model':MODEL,'dimensions':DIMENSIONS,'batch_size':BATCH,
        'query_instruction':cache['metadata']['query_instruction'],
        'corpus_fingerprint':corpus.fingerprint,'document_fingerprint':document_fingerprint,
        'query_fingerprint':query_fingerprint(cases),'dev_sha256':manifest['files']['dev.jsonl']['sha256'],
        'baseline_scores_sha256':sha(args.baseline_scores),'baseline_snapshot_sha256':sha(args.baseline_snapshot),
        'baseline_snapshot_path':args.baseline_snapshot.resolve().relative_to(ROOT).as_posix(),
        'baseline_responses_sha256':sha(args.baseline_responses),
        'case_ids':old_plan['case_ids'],'baseline_rerank':baseline,
        'embedding_request_cap':350,'rerank_request_cap':100,
        'embedding_planned_requests':math.ceil(len(docs)/BATCH)+math.ceil(len(cases)/BATCH),
        'rerank_model':'Qwen/Qwen3-Reranker-4B','rerank_instruction':ENGLISH_INSTRUCTION,
        'pool_size':40,'fusion':BASELINE,
        'payload':{'document_count':len(docs),'dev_query_count':len(cases),'rerank_query_count':100,
                   'embedding_document_characters':sum(len(text) for _,text in docs),
                   'embedding_query_characters':sum(len(case['query']) for case in cases)},
        'scope':'New separate 4B index; retain 0.6B. Fixed dev/pilot, no frozen test, no query filtering, no gold input.'}
    args.plan.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.plan,plan)
    print(json.dumps({key:plan[key] for key in ('model','dimensions','payload','embedding_planned_requests',
                                               'embedding_request_cap','rerank_request_cap')},ensure_ascii=False))


def validate(plan,cache):
    corpus,cases,manifest,docs=inputs()
    validate_cache(cache)
    if (plan['version']!='careguide-embedding-ablation-v1' or plan['model']!=MODEL
            or plan['dimensions']!=DIMENSIONS or plan['batch_size']!=BATCH
            or plan['corpus_fingerprint']!=corpus.fingerprint
            or plan['query_fingerprint']!=query_fingerprint(cases)
            or plan['dev_sha256']!=manifest['files']['dev.jsonl']['sha256']
            or plan['document_fingerprint']!=text_hash(json.dumps([(key,text_hash(text)) for key,text in docs]))
            or plan['fusion']!=BASELINE or plan['pool_size']!=40
            or plan['rerank_model']!='Qwen/Qwen3-Reranker-4B' or plan['rerank_instruction']!=ENGLISH_INSTRUCTION
            or plan['query_instruction']!=cache['metadata']['query_instruction']):
        raise ValueError('Embedding experiment contract changed')
    from scripts.evaluate_rag_candidate_pools import SAMPLE_SEED
    _,validation=partition_rows(cache['rows'])
    expected=sorted(validation,key=lambda row:hashlib.sha256((SAMPLE_SEED+row['id']).encode()).hexdigest())[:100]
    if plan['case_ids']!=[r['id'] for r in expected]:
        raise ValueError('Must retain the original fixed query sample')
    return corpus,cases,docs


def require_endpoint(value):
    url=urlparse(value.rstrip('/'))
    if ((url.scheme,url.hostname,url.path,url.port)!=('https','api.siliconflow.cn','/v1',None)
            or url.username is not None or url.password is not None or url.query or url.fragment):
        raise ValueError('Only explicitly scoped SiliconFlow endpoint allowed')


def experiment_dir(path):
    work=path.resolve()
    allowed=(ROOT/'data/source_cache').resolve()
    if allowed not in work.parents or 'qwen-embedding-baselines' in work.parts:
        raise ValueError('Use a separate data/source_cache subdirectory outside the preserved baselines')
    return work


def verify_preserved_index(plan):
    path=ROOT/plan['baseline_snapshot_path']
    if sha(path)!=plan['baseline_snapshot_sha256']:
        raise ValueError('Baseline snapshot manifest changed')
    manifest=read_json(path)
    if tree_hashes(Path(manifest['destination']))!=manifest['files']:
        raise ValueError('Preserved baseline index files changed')


def cache_vectors(db,rows,kind):
    saved={key:(digest,blob) for key,digest,blob in db.execute(
        'select id,text_sha256,vector from vectors where kind=?',(kind,))}
    if set(saved)-{key for key,_ in rows}:
        raise ValueError('Foreign vectors in checkpoint')
    for key,text in rows:
        if key in saved:
            digest,blob=saved[key]
            vector=np.frombuffer(blob,dtype='<f4')
            if digest!=text_hash(text) or len(vector)!=DIMENSIONS or not np.isfinite(vector).all():
                raise ValueError('Checkpoint vector/text invalid')
    return saved


def embed(plan,plan_path,work,corpus,cases,docs):
    identity={'plan_sha256':sha(plan_path),'model':MODEL,'dimensions':DIMENSIONS,
              'endpoint':'https://api.siliconflow.cn/v1/embeddings'}
    budget=AttemptLedger(work/'embedding-requests.json',identity,350)
    base=os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL','')
    require_endpoint(base)
    usage=[]
    def observe(response):
        value=getattr(response,'usage',None)
        usage.append(int(getattr(value,'prompt_tokens',0)) if value else None)
    provider=None
    with closing(sqlite3.connect(work/'vectors.sqlite3')) as db:
        db.execute('create table if not exists config (identity text not null)')
        db.execute('create table if not exists vectors (kind text,id text,text_sha256 text,vector blob,primary key(kind,id))')
        stored=db.execute('select identity from config').fetchone()
        if stored and json.loads(stored[0])!=identity:
            raise ValueError('Embedding checkpoint belongs to another plan')
        if not stored:
            db.execute('insert into config values (?)',(json.dumps(identity),));db.commit()
        groups=[('doc',docs),('query',[(case['id'],case['query']) for case in cases])]
        missing=[]
        for kind,items in groups:
            existing=cache_vectors(db,items,kind)
            absent=[(key,text) for key,text in items if key not in existing]
            missing.append((kind,absent))
        needed=sum(math.ceil(len(values)/BATCH) for _,values in missing)
        if needed+len(budget.data['attempts'])>350:
            raise ValueError('Insufficient remaining embedding budget; no calls sent')
        try:
            if needed:
                provider=OpenAICompatibleEmbeddingFunction(api_key=os.environ.get('CAREGUIDE_EMBEDDING_API_KEY',''),
                    base_url=base,model=MODEL,dimensions=DIMENSIONS,query_instruction=plan['query_instruction'],
                    batch_size=BATCH,max_retries=0,timeout=90,response_observer=observe)
            for kind,items in missing:
                for start in range(0,len(items),BATCH):
                    batch=items[start:start+BATCH]
                    budget.reserve({'case_id':kind+':'+','.join(key for key,_ in batch),'arm':'embedding_'+kind})
                    tick=time.perf_counter()
                    try:
                        vectors=(provider.embed_documents([text for _,text in batch]) if kind=='doc'
                                 else provider.embed_queries([text for _,text in batch]))
                    except Exception as error:
                        raise RuntimeError(f'Embedding attempt failed ({type(error).__name__}); budget consumed') from None
                    for (key,text),vector in zip(batch,vectors):
                        db.execute('insert into vectors values (?,?,?,?)',
                            (kind,key,text_hash(text),np.asarray(vector,dtype='<f4').tobytes()))
                    db.commit()
                    budget.data['attempts'][-1].update({'seconds':time.perf_counter()-tick,'input_tokens':usage[-1]})
                    atomic_json(budget.path,budget.data)
                    if len(budget.data['attempts'])%10==0 or start+BATCH>=len(items):
                        print(json.dumps({'stage':'embedding_'+kind,'completed_rows':start+len(batch),
                                          'attempted':len(budget.data['attempts'])}),flush=True)
        finally:
            if provider:provider.close()


def build_scores(plan,work,corpus,cases,docs,baseline):
    import chromadb
    identity=SimpleNamespace(model=MODEL,dimensions=DIMENSIONS)
    with closing(sqlite3.connect((work/'vectors.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)) as db:
        document_vectors=cache_vectors(db,docs,'doc')
        query_rows=[(case['id'],case['query']) for case in cases]
        query_vectors=cache_vectors(db,query_rows,'query')
    if len(document_vectors)!=len(docs) or len(query_vectors)!=len(cases):
        raise ValueError('Vector checkpoint is not complete')
    client=chromadb.PersistentClient(path=str(work/'chroma'),settings=chromadb.Settings(anonymized_telemetry=False))
    name=versioned_collection_name(corpus)
    metadata={'embedding_model':MODEL,'embedding_dimensions':DIMENSIONS,'hnsw:space':'cosine',
              'corpus_fingerprint':corpus.fingerprint,'reuse_complete':True,
              'embedding_text_sha256':plan['document_fingerprint']}
    names={c.name for c in client.list_collections()}
    published=False
    if name in names:
        collection=client.get_collection(name,embedding_function=None)
        actual=collection.metadata or {}
        if any(actual.get(key)!=value for key,value in metadata.items() if key!='reuse_complete'):
            raise ValueError('New index exists with a different identity')
        if actual.get('reuse_complete') is True:
            published=True
        else:
            # Preserve the incomplete experiment collection; never touch a baseline.
            import uuid
            collection.modify(name=name+'_unpublished_'+uuid.uuid4().hex[:8])
    if not published:
        staging=name+'_building_4b1024'
        names={c.name for c in client.list_collections()}
        if staging in names:
            collection=client.get_collection(staging,embedding_function=None)
            if any((collection.metadata or {}).get(key)!=value for key,value in metadata.items()):
                raise ValueError('Staging index belongs to a different build')
        else:
            # Chroma 0.5 rejects metadata updates containing hnsw:space, even if
            # unchanged. Keep the final metadata on an undiscoverable staging
            # name, and publish by rename only after complete ID verification.
            collection=client.create_collection(staging,metadata=metadata,embedding_function=None)
    present=set(collection.get(include=[])['ids'])
    if present-{key for key,_ in docs}:
        raise ValueError('Foreign documents in new index')
    by_id={record.id:record for record in corpus.records}
    remaining=[key for key,_ in docs if key not in present]
    for start in range(0,len(remaining),256):
        keys=remaining[start:start+256]
        collection.add(ids=keys,embeddings=[np.frombuffer(document_vectors[key][1],dtype='<f4').tolist() for key in keys],
            documents=[by_id[key].content for key in keys],metadatas=[{'record_id':key,'source_id':by_id[key].source_id,
                                                                    'kind':by_id[key].kind.value} for key in keys])
    if set(collection.get(include=[])['ids'])!={key for key,_ in docs}:
        raise ValueError('Staging index is incomplete; not published')
    if not published:
        collection.modify(name=name)
    MedicalKnowledgeBase(corpus,collection=collection,embedding=identity,mode='dense')
    rows=[]
    originals={row['id']:row for row in baseline['rows']}
    for start in range(0,len(cases),BATCH):
        batch=cases[start:start+BATCH]
        result=collection.query(query_embeddings=[np.frombuffer(query_vectors[c['id']][1],dtype='<f4').tolist()
                                                 for c in batch],n_results=100,include=['distances'])
        for case,ids,distances in zip(batch,result['ids'],result['distances']):
            rows.append({'id':case['id'],'gold_record_id':case['gold_record_id'],'streams':{
                'bm25':originals[case['id']]['streams']['bm25'],
                'dense':[(key,1-float(distance)) for key,distance in zip(ids,distances)]}})
    cache={'metadata':{**baseline['metadata'],'embedding_model':MODEL,'embedding_dimensions':DIMENSIONS,
                       'external_requests':len(read_json(work/'embedding-requests.json')['attempts']),
                       'baseline_scores_sha256':plan['baseline_scores_sha256']},'rows':rows}
    validate_cache(cache)
    atomic_json(work/'scores.json',cache)
    return cache


def evaluate(plan,work,old,new,plan_path,allow_remote,response_cache=None,output_path=None):
    validate_cache(new)
    if (new['metadata'].get('embedding_model')!=MODEL or new['metadata'].get('embedding_dimensions')!=DIMENSIONS
            or new['metadata'].get('query_instruction')!=plan['query_instruction']):
        raise ValueError('New score cache model/dimensions/instruction mismatch')
    rows={row['id']:row for row in new['rows']}
    checkpoint=response_cache or work/'rerank-responses.json'
    identity={'plan_sha256':sha(plan_path),'model':plan['rerank_model'],'instruction':plan['rerank_instruction'],
              'endpoint':'https://api.siliconflow.cn/v1/rerank'}
    saved=read_json(checkpoint) if checkpoint.exists() else {'identity':identity,'completed':[]}
    if saved['identity']!=identity:raise ValueError('Rerank checkpoint identity mismatch')
    done={r['case_id'] for r in saved['completed']}
    missing=[key for key in plan['case_ids'] if key not in done]
    if missing and not allow_remote:raise ValueError('Missing actual rerank responses; explicit remote authorization required')
    if missing:
        budget=AttemptLedger(work/'rerank-requests.json',identity,100)
        if len(budget.data['attempts'])+len(missing)>100:raise ValueError('Rerank request cap cannot complete the plan')
        base=os.environ.get('CAREGUIDE_RERANK_BASE_URL') or os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL','')
        require_endpoint(base)
        adapter=OpenAICompatibleReranker(api_key=os.environ.get('CAREGUIDE_RERANK_API_KEY')
            or os.environ.get('CAREGUIDE_EMBEDDING_API_KEY',''),base_url=base,model=plan['rerank_model'],
            instruction=plan['rerank_instruction'],max_retries=0,timeout=90)
        corpus,cases,_,_=inputs()
        queries={case['id']:case['query'] for case in cases}
        records={r.id:r for r in corpus.records}
        for key in missing:
            selected=rank(rows[key],BASELINE)[:40]
            budget.reserve({'case_id':key,'arm':'embedding_4b_rerank_4b40'})
            tick=time.perf_counter()
            try:ids,scores,meta=adapter.rerank(queries[key],[(doc,records[doc].content) for doc in selected],top_n=len(selected))
            except Exception as error:raise RuntimeError(f'Rerank attempt failed ({type(error).__name__}); budget consumed') from None
            if any(not math.isfinite(score) for score in scores):raise ValueError('Nonfinite rerank response')
            saved['completed'].append({'case_id':key,'candidate_ids':selected,'ids':ids,'scores':scores,
                'seconds':time.perf_counter()-tick,'input_tokens':(int(meta['tokens']['input_tokens'])
                    if 'input_tokens' in (meta.get('tokens') or {}) else None)})
            atomic_json(checkpoint,saved)
            if len(saved['completed'])%10==0:print(json.dumps({'stage':'rerank','completed':len(saved['completed'])}),flush=True)
    by_id={row['case_id']:row for row in saved['completed']}
    if len(by_id)!=100 or len(saved['completed'])!=100:raise ValueError('Incomplete/duplicate rerank sample')
    before={row['case_id']:row for row in plan['baseline_rerank']}
    gold=[rows[key]['gold_record_id'] for key in plan['case_ids']]
    for key in plan['case_ids']:
        response=by_id[key];pool=rank(rows[key],BASELINE)[:40]
        if (response['candidate_ids']!=pool or len(response['ids'])!=len(pool)
                or set(response['ids'])!=set(pool) or len(response['scores'])!=len(pool)):
            raise ValueError('Actual response/pool mismatch')
    rankings={name:[source[key]['ids'][:10] for key in plan['case_ids']]
              for name,source in [('embedding_06b',before),('embedding_4b',by_id)]}
    changes=[int(t in a)-int(t in b) for t,b,a in zip(gold,rankings['embedding_06b'],rankings['embedding_4b'])]
    partitions={}
    for name,cache in [('embedding_06b',old),('embedding_4b',new)]:
        tune,valid=partition_rows(cache['rows'])
        partitions[name]={}
        for label,selected in [('tuning500',tune),('validation500',valid),('all_dev',cache['rows'])]:
            ranked=[rank(row,BASELINE) for row in selected]
            targets=[row['gold_record_id'] for row in selected]
            partitions[name][label]={**ranking_metrics(ranked,targets,cutoff=10),
                **{f'coverage_{k}':sum(t in r[:k] for t,r in zip(targets,ranked))/len(selected) for k in (20,40,100)},
                'dense_top100':sum(row['gold_record_id'] in dict(row['streams']['dense']) for row in selected)/len(selected),
                'union_top100_each':sum(any(row['gold_record_id'] in dict(v) for v in row['streams'].values()) for row in selected)/len(selected)}
    result={'evaluation':'0.6B vs 4B embedding, same 4B reranker and Top40','frozen_test_run':False,
        'pre_rerank':partitions,'pilot':{name:ranking_metrics(ranking,gold,cutoff=10) for name,ranking in rankings.items()},
        'paired_pilot':{**paired_interval(changes),'gained':changes.count(1),'lost':changes.count(-1)},
        'pilot_candidate_coverage':{name:sum(t in source[key]['candidate_ids'] for key,t in zip(plan['case_ids'],gold))/100
                                    for name,source in [('embedding_06b',before),('embedding_4b',by_id)]},
        'source_plan_sha256':sha(plan_path),'scope':plan['scope']}
    target=output_path or work/'result.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(target,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run','replay'))
    for name in ('plan','baseline-scores','baseline-plan','baseline-responses','baseline-snapshot','work-dir','config-file',
                 'new-scores','responses','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--allow-remote',action='store_true')
    parser.add_argument('--max-embedding-requests',type=int)
    parser.add_argument('--max-rerank-requests',type=int)
    args=parser.parse_args()
    if args.plan is None or args.baseline_scores is None:parser.error('Explicit plan and baseline scores required')
    if args.mode=='prepare':
        if not all((args.baseline_plan,args.baseline_responses,args.baseline_snapshot)):parser.error('Verified baseline sources required')
        prepare(args);return
    plan,baseline=read_json(args.plan),read_json(args.baseline_scores)
    if sha(args.baseline_scores)!=plan['baseline_scores_sha256']:raise ValueError('Baseline score checksum changed')
    corpus,cases,docs=validate(plan,baseline)
    if args.mode=='replay':
        if args.new_scores is not None and args.responses is not None and args.output is not None:
            evaluate(plan,None,baseline,read_json(args.new_scores),args.plan,False,
                     response_cache=args.responses,output_path=args.output)
        elif args.work_dir is not None:
            work=experiment_dir(args.work_dir)
            evaluate(plan,work,baseline,read_json(work/'scores.json'),args.plan,False)
        else:parser.error('Replay requires explicit public scores/responses/output, or the local experiment work-dir')
        return
    if args.work_dir is None:parser.error('Separate experiment work-dir required')
    work=experiment_dir(args.work_dir)
    verify_preserved_index(plan)
    if (not args.allow_remote or args.config_file is None or args.max_embedding_requests!=350 or args.max_rerank_requests!=100):
        print(json.dumps({'mode':'offline dry-run','planned_embeddings':plan['embedding_planned_requests'],
                          'embedding_cap':350,'rerank_cap':100,'new_index':str(work/'chroma')}));return
    work.mkdir(parents=True,exist_ok=True)
    binding=work/'plan-binding.json'
    if binding.exists() and read_json(binding)!= {'plan_sha256':sha(args.plan)}:raise ValueError('Work directory belongs to another plan')
    if not binding.exists() and any(work.iterdir()):raise ValueError('Work directory is not empty or bound to this experiment')
    if not binding.exists():atomic_json(binding,{'plan_sha256':sha(args.plan)})
    load_config(args.config_file)
    embed(plan,args.plan,work,corpus,cases,docs)
    new=build_scores(plan,work,corpus,cases,docs,baseline)
    evaluate(plan,work,baseline,new,args.plan,True)
    verify_preserved_index(plan)


if __name__=='__main__':
    main()
