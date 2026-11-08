"""Controlled dev-only enrichment ablation with cached query vectors.

Generate a separate contextual document index; retain original answer text for
reranking and citations. The original 0.6B model, 1024 dimensions, BM25 tokenizer,
RRF(4:1,k=60), Top100/leg, Top40 reranker inputs and fixed 100-query pilot remain
unchanged. No query embedding calls, frozen test, tools or runtime promotion.
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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from careguide.retrieval.bm25_index import BM25Index
from careguide.retrieval.embedding_provider import OpenAICompatibleEmbeddingFunction
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, versioned_collection_name
from careguide.retrieval.reranker import OpenAICompatibleReranker
from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json, query_fingerprint
from scripts.compare_rag_embeddings import require_endpoint, text_hash
from scripts.compare_rag_rerank_models import ENGLISH_INSTRUCTION
from scripts.contextualize_rag_corpus import (digest, grounded_prefix, ordered_records, prepare,
    source_payload, validate_prefixes, work_directory)
from scripts.evaluate_rag_candidate_pools import BASELINE, SAMPLE_SEED, partition_rows, rank
from scripts.evaluate_rag_score_fusion import validate_cache
from scripts.private_config import load_config
from scripts.rerank_rag_candidate_plan import AttemptLedger, paired_interval, sha
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, ROOT, load_cases

MODEL = 'Qwen/Qwen3-Embedding-0.6B'
DIMENSIONS = 1024
BATCH = 32
SCORES = ROOT/'data/eval/cache/rag-recall-scores-dev-20261105.json.gz'
BASELINE_RESPONSES = ROOT/'data/eval/cache/rag-reranker-model-dev-20261106.json.gz'
BASELINE_PLAN = ROOT/'data/eval/cache/rag-reranker-model-plan-dev-20261106.json'
QUERY_CACHE = ROOT/'data/source_cache/qwen-embedding-baselines/06b-1024-dev-query-vectors-20261106.json'


def write_gzip(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(gzip.compress(json.dumps(value, ensure_ascii=False).encode(), mtime=0))
    temporary.replace(path)


def inputs(work):
    generation_plan, _ = prepare(work)
    audit = read_json(work/'audit.json')
    if audit['plan_sha256'] != digest(generation_plan) or audit['decision'] != 'proceed':
        raise ValueError('Successful pilot content audit required')
    corpus = Corpus(CORPUS_DIR)
    saved = read_json(work/'prefixes.json')
    prefixes = {r['id']:r for r in saved['records']}
    if (len(prefixes) != len(corpus.records) or len(saved['records']) != len(prefixes)
            or set(prefixes) != {r.id for r in corpus.records}):
        raise ValueError('Full corpus prefixes required for a fair full-corpus comparison')
    texts = []
    rejected = []
    for record in corpus.records:
        row = prefixes[record.id]
        if row['source_sha256'] != digest(source_payload(record)):
            raise ValueError('Generated prefix source changed')
        if row.get('rejected_reason'):
            if row['context'] or row['anchors']:
                raise ValueError('Rejected context must be empty')
            rejected.append(record.id)
        else:
            validate_prefixes([source_payload(record)], {'records':[{k:row[k] for k in ('id','context','anchors')}]})
            expected = grounded_prefix(source_payload(record),{k:row[k] for k in ('id','terms','aspects')})
            if any(row[k] != expected[k] for k in expected):
                raise ValueError('Prefix is not the literal-subject deterministic template')
        base = MedicalKnowledgeBase.retrieval_text(record)
        texts.append((record.id, row['context']+'\n'+base if row['context'] else base))
    cases, manifest = load_cases('dev')
    old = read_json(SCORES)
    validate_cache(old)
    if (old['metadata']['corpus_fingerprint'] != corpus.fingerprint
            or old['metadata']['query_fingerprint'] != query_fingerprint(cases)
            or old['metadata']['embedding_model'] != MODEL
            or old['metadata']['embedding_dimensions'] != DIMENSIONS):
        raise ValueError('Baseline identity mismatch')
    queries = read_json(QUERY_CACHE)
    if queries['identity'] != {'query_fingerprint':query_fingerprint(cases), 'model':MODEL,
            'dimensions':DIMENSIONS, 'instruction':old['metadata']['query_instruction']}:
        raise ValueError('Cached query vector identity mismatch')
    vectors = np.asarray(queries['vectors'],dtype='<f4')
    if vectors.shape != (len(cases),DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError('Invalid cached query vectors')
    responses = read_json(BASELINE_RESPONSES)
    baseline_plan = read_json(BASELINE_PLAN)
    if (responses['identity']['plan_sha256'] != sha(BASELINE_PLAN)
            or baseline_plan['arms']['english_4b40'] != {'model':'Qwen/Qwen3-Reranker-4B',
                'instruction':ENGLISH_INSTRUCTION,'pool_size':40,'cached':False}):
        raise ValueError('Baseline rerank model/instruction identity mismatch')
    baseline = {r['case_id']:r for r in responses['completed'] if r['arm'] == 'english_4b40'}
    _, validation = partition_rows(old['rows'])
    selected = sorted(validation, key=lambda r:hashlib.sha256((SAMPLE_SEED+r['id']).encode()).hexdigest())[:100]
    case_ids = [r['id'] for r in selected]
    if len(baseline) != 100 or set(baseline) != set(case_ids):
        raise ValueError('Fixed 100-query baseline incomplete')
    for row in selected:
        if baseline[row['id']]['candidate_ids'] != rank(row, BASELINE)[:40]:
            raise ValueError('Baseline rerank pool mismatch')
    plan = {'version':'careguide-contextual-retrieval-ablation-v1',
        'generation_plan_sha256':digest(generation_plan), 'prefixes_sha256':sha(work/'prefixes.json'),
        'corpus_fingerprint':corpus.fingerprint, 'dev_sha256':manifest['files']['dev.jsonl']['sha256'],
        'document_fingerprint':digest([(key,text_hash(text)) for key,text in texts]),
        'baseline_scores_sha256':sha(SCORES), 'baseline_responses_sha256':sha(BASELINE_RESPONSES),
        'query_vectors_sha256':sha(QUERY_CACHE), 'query_instruction':old['metadata']['query_instruction'],
        'case_ids':case_ids, 'model':MODEL, 'dimensions':DIMENSIONS, 'fusion':BASELINE,
        'rerank_model':'Qwen/Qwen3-Reranker-4B', 'rerank_instruction':ENGLISH_INSTRUCTION,
        'rerank_text':'original_content', 'rerank_candidates':40, 'candidate_k':100,
        'maximum_embedding_requests':318, 'maximum_rerank_requests':105,
        'planned_embedding_requests':math.ceil(len(texts)/BATCH), 'planned_rerank_requests':100,
        'rejected_prefixes_original_fallback':rejected, 'frozen_test_run':False}
    path = work/'evaluation-plan.json'
    if path.exists() and read_json(path) != plan:
        raise ValueError('Evaluation plan changed')
    if not path.exists():
        atomic_json(path, plan)
    return plan, corpus, cases, texts, vectors, old, baseline


def embed(work, plan, texts):
    base = os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL','')
    require_endpoint(base)
    identity = {'plan_sha256':digest(plan),'model':MODEL,'dimensions':DIMENSIONS,
                'endpoint':'https://api.siliconflow.cn/v1/embeddings'}
    ledger = AttemptLedger(work/'embedding-requests.json',identity,plan['maximum_embedding_requests'])
    def encode_batch(batch):
        usage = []
        def observe(response):
            usage.append(response.usage.prompt_tokens if response.usage else None)
        provider = OpenAICompatibleEmbeddingFunction(
            api_key=os.environ.get('CAREGUIDE_EMBEDDING_API_KEY',''),base_url=base,
            model=MODEL,dimensions=DIMENSIONS,batch_size=BATCH,max_retries=0,timeout=90,
            query_instruction=plan['query_instruction'],response_observer=observe)
        tick = time.perf_counter()
        try:
            result = provider.embed_documents([text for _,text in batch])
            if len(result) != len(batch):
                raise ValueError('Incomplete provider batch')
            return result,usage[-1],time.perf_counter()-tick
        finally:
            provider.close()
    with closing(sqlite3.connect(work/'vectors.sqlite3')) as db:
        db.execute('create table if not exists identity (value text)')
        db.execute('create table if not exists vectors (id text primary key, text_sha256 text, vector blob)')
        row = db.execute('select value from identity').fetchone()
        if row and json.loads(row[0]) != identity:
            raise ValueError('Vector checkpoint belongs to another experiment')
        if not row:
            db.execute('insert into identity values (?)',(json.dumps(identity),)); db.commit()
        saved = {key:(h,blob) for key,h,blob in db.execute('select * from vectors')}
        if set(saved)-{key for key,_ in texts}:
            raise ValueError('Foreign cached vectors')
        for key,text in texts:
            if key in saved:
                h,blob = saved[key]
                value = np.frombuffer(blob,dtype='<f4')
                if h != text_hash(text) or len(value) != DIMENSIONS or not np.isfinite(value).all():
                    raise ValueError('Changed input or corrupt cached vector')
        missing = [(key,text) for key,text in texts if key not in saved]
        if len(ledger.data['attempts'])+math.ceil(len(missing)/BATCH) > ledger.maximum:
            raise ValueError('Insufficient remaining embedding request budget')
        batches = [missing[start:start+BATCH] for start in range(0,len(missing),BATCH)]
        pending, next_batch, completed, failure = {}, 0, len(saved), None
        with ThreadPoolExecutor(max_workers=4) as executor:
            while pending or (next_batch<len(batches) and failure is None):
                while failure is None and len(pending)<4 and next_batch<len(batches):
                    batch = batches[next_batch]
                    ledger.reserve({'case_id':','.join(key for key,_ in batch),'arm':'contextual_documents'})
                    attempt = ledger.data['attempts'][-1]
                    attempt['status'] = 'reserved'
                    atomic_json(ledger.path,ledger.data)
                    pending[executor.submit(encode_batch,batch)] = (batch,attempt)
                    next_batch += 1
                finished,_ = wait(pending,return_when=FIRST_COMPLETED)
                for future in finished:
                    batch,attempt = pending.pop(future)
                    try:
                        result,tokens,seconds = future.result()
                    except Exception as error:
                        failure = type(error).__name__
                        attempt.update(status='failed',error_type=failure)
                        atomic_json(ledger.path,ledger.data)
                        continue
                    # Only the parent writes SQLite/checkpoints. On failure no
                    # new calls start; already reserved in-flight successes drain.
                    for (key,text),vector in zip(batch,result):
                        db.execute('insert into vectors values (?,?,?)',
                                   (key,text_hash(text),np.asarray(vector,dtype='<f4').tobytes()))
                    db.commit()
                    completed += len(batch)
                    attempt.update(status='completed',seconds=seconds,input_tokens=tokens)
                    atomic_json(ledger.path,ledger.data)
                    if (completed-len(saved))//BATCH%10 == 0 or completed == len(texts):
                        print(json.dumps({'stage':'embedding','completed':completed,
                            'attempts':len(ledger.data['attempts'])}),flush=True)
        if failure:
            raise RuntimeError('Embedding stopped ('+failure+'); no automatic retry')


def build_scores(work, plan, corpus, cases, texts, queries, old):
    import chromadb
    with closing(sqlite3.connect((work/'vectors.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)) as db:
        vectors = {key:(h,blob) for key,h,blob in db.execute('select * from vectors')}
    if set(vectors) != {key for key,_ in texts}:
        raise ValueError('Document embedding checkpoint incomplete')
    matrix = []
    for key,text in texts:
        h,blob = vectors[key]
        value = np.frombuffer(blob,dtype='<f4')
        if h != text_hash(text) or len(value) != DIMENSIONS or not np.isfinite(value).all():
            raise ValueError('Document vector provenance invalid')
        matrix.append(value.tolist())
    client = chromadb.PersistentClient(path=str(work/'chroma'),settings=chromadb.Settings(anonymized_telemetry=False))
    name = versioned_collection_name(corpus)+'_contextual'
    metadata = {'corpus_fingerprint':corpus.fingerprint,'embedding_model':MODEL,
        'embedding_dimensions':DIMENSIONS,'hnsw:space':'cosine',
        'contextual_document_fingerprint':plan['document_fingerprint']}
    if name in {c.name for c in client.list_collections()}:
        collection = client.get_collection(name,embedding_function=None)
        if collection.metadata != metadata:
            raise ValueError('Contextual index identity changed')
    else:
        collection = client.create_collection(name,embedding_function=None,metadata=metadata)
    present = set(collection.get(include=[])['ids'])
    if present-{key for key,_ in texts}:
        raise ValueError('Foreign index documents')
    rows = {r.id:r for r in corpus.records}
    for start in range(0,len(texts),256):
        selected = [(i,key,text) for i,(key,text) in enumerate(texts[start:start+256],start) if key not in present]
        if selected:
            collection.add(ids=[key for _,key,_ in selected],
                embeddings=[matrix[i] for i,_,_ in selected],documents=[rows[key].content for _,key,_ in selected],
                metadatas=[{'record_id':key,'source_id':rows[key].source_id} for _,key,_ in selected])
    if set(collection.get(include=[])['ids']) != set(rows):
        raise ValueError('New index incomplete')
    lexical = BM25Index()
    lexical.upsert({'chunk_id':key,'content':rows[key].content,'retrieval_text':text,'metadata':{}} for key,text in texts)
    scored = []
    for start in range(0,len(cases),BATCH):
        batch = cases[start:start+BATCH]
        result = collection.query(query_embeddings=queries[start:start+len(batch)].tolist(),n_results=100,
                                  include=['distances'])
        for case,ids,distances in zip(batch,result['ids'],result['distances']):
            scored.append({'id':case['id'],'gold_record_id':case['gold_record_id'],'streams':{
                'dense':[(key,1-float(d)) for key,d in zip(ids,distances)],
                'bm25':[(h.chunk_id,h.score) for h in lexical.search(case['query'],top_k=100)]}})
        if start%128 == 0:
            print(json.dumps({'stage':'local_retrieval','completed':len(scored)}),flush=True)
    attempted = len(read_json(work/'embedding-requests.json')['attempts'])
    cache = {'metadata':{**old['metadata'],'document_fingerprint':plan['document_fingerprint'],
        'external_requests':attempted,'document_embedding_requests':attempted,
        'query_embedding_requests':0,'scope':'Full-corpus enriched document index, same original query vectors'},'rows':scored}
    validate_cache(cache)
    write_gzip(work/'scores.json.gz',cache)
    return cache


def stage_metrics(rows):
    rankings = [rank(row,BASELINE) for row in rows]
    targets = [row['gold_record_id'] for row in rows]
    return {**ranking_metrics(rankings,targets,cutoff=10),
        **{f'coverage_{k}':sum(t in r[:k] for t,r in zip(targets,rankings))/len(rows) for k in (20,40,100)},
        **{name+'_top100':sum(row['gold_record_id'] in dict(row['streams'][name]) for row in rows)/len(rows)
           for name in ('dense','bm25')},
        'union_top100_each':sum(any(row['gold_record_id'] in dict(s) for s in row['streams'].values())
                               for row in rows)/len(rows)}


def summarize_stages(work, plan, old, new):
    results = {}
    # Offline 2x2 controls isolate the lexical and dense effects without
    # changing weights or making additional model requests.
    originals = {row['id']:row for row in old['rows']}
    controls = []
    for label,changed_leg in [('contextual_bm25_only','bm25'),('contextual_dense_only','dense')]:
        controls.append((label,{'rows':[{
            **row,'streams':{name:(row['streams'][name] if name==changed_leg
                                  else originals[row['id']]['streams'][name])
                              for name in ('bm25','dense')}} for row in new['rows']]}))
    for name,cache in [('baseline',old),('contextual',new),*controls]:
        tune,validation = partition_rows(cache['rows'])
        results[name] = {k:stage_metrics(v) for k,v in [('tuning500',tune),('validation500',validation),
            ('all_dev1000',cache['rows']),('fixed_pilot100',[r for r in cache['rows'] if r['id'] in plan['case_ids']])]}
    atomic_json(work/'stage-results.json',results)
    print(json.dumps(results,ensure_ascii=False),flush=True)


def rerank(work, plan, cases, corpus, new, *, allow_remote):
    path = work/'rerank-responses.json'
    identity = {'plan_sha256':digest(plan),'scores_sha256':sha(work/'scores.json.gz'),
        'model':plan['rerank_model'],'instruction':plan['rerank_instruction'],
        'endpoint':'https://api.siliconflow.cn/v1/rerank','text':'original_content'}
    saved = read_json(path) if path.exists() else {'identity':identity,'completed':[]}
    if saved['identity'] != identity:
        raise ValueError('Response identity mismatch')
    done = {r['case_id']:r for r in saved['completed']}
    if len(done) != len(saved['completed']) or set(done)-set(plan['case_ids']):
        raise ValueError('Foreign/duplicate rerank responses')
    missing = [key for key in plan['case_ids'] if key not in done]
    if missing and not allow_remote:
        raise ValueError('Real responses missing; no result fabricated from a different candidate pool')
    if not missing:
        return done
    base = os.environ.get('CAREGUIDE_RERANK_BASE_URL') or os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL','')
    require_endpoint(base)
    ledger = AttemptLedger(work/'rerank-requests.json',identity,plan['maximum_rerank_requests'])
    if len(ledger.data['attempts'])+len(missing) > ledger.maximum:
        raise ValueError('Insufficient remaining rerank budget')
    adapter = OpenAICompatibleReranker(api_key=os.environ.get('CAREGUIDE_RERANK_API_KEY')
        or os.environ.get('CAREGUIDE_EMBEDDING_API_KEY',''),base_url=base,model=plan['rerank_model'],
        instruction=plan['rerank_instruction'],max_retries=0,timeout=90)
    rows = {r['id']:r for r in new['rows']}
    queries = {r['id']:r['query'] for r in cases}
    contents = {r.id:r.content for r in corpus.records}
    try:
        for key in missing:
            pool = rank(rows[key],BASELINE)[:40]
            ledger.reserve({'case_id':key,'arm':'contextual_original_rerank40'})
            tick = time.perf_counter()
            try:
                ids,scores,meta = adapter.rerank(queries[key],[(doc,contents[doc]) for doc in pool],top_n=len(pool))
            except Exception as error:
                raise RuntimeError('Rerank stopped ('+type(error).__name__+'); no automatic retry') from None
            if set(ids) != set(pool) or len(ids) != len(pool) or any(not math.isfinite(s) for s in scores):
                raise ValueError('Invalid reranker output')
            saved['completed'].append({'case_id':key,'candidate_ids':pool,'ids':ids,'scores':scores,
                'seconds':time.perf_counter()-tick,'input_tokens':(meta.get('tokens') or {}).get('input_tokens')})
            atomic_json(path,saved)
            if len(saved['completed'])%10 == 0:
                print(json.dumps({'stage':'rerank','completed':len(saved['completed'])}),flush=True)
    finally:
        # The adapter owns a short-lived HTTP client inside each request.
        pass
    return {r['case_id']:r for r in saved['completed']}


def final_metrics(plan, new, baseline, actual):
    if set(actual) != set(plan['case_ids']):
        raise ValueError('Incomplete final rerank sample')
    rows = {r['id']:r for r in new['rows']}
    for key,row in actual.items():
        pool = rank(rows[key],BASELINE)[:40]
        if row['candidate_ids'] != pool or set(row['ids']) != set(pool) or len(row['ids']) != len(pool):
            raise ValueError('Candidate/response mismatch')
    targets = [rows[key]['gold_record_id'] for key in plan['case_ids']]
    before = [baseline[key]['ids'][:10] for key in plan['case_ids']]
    after = [actual[key]['ids'][:10] for key in plan['case_ids']]
    changes = [int(t in a)-int(t in b) for t,a,b in zip(targets,after,before)]
    return {'evaluation':'Grounded source-only enrichment, same 0.6B/1024 embedding + 4B original-content rerank Top40',
        'frozen_test_run':False,'baseline':ranking_metrics(before,targets,cutoff=10),
        'contextual':ranking_metrics(after,targets,cutoff=10),
        'paired':{**paired_interval(changes),'gained':changes.count(1),'lost':changes.count(-1)},
        'case_results':[{'id':key,'gold_record_id':target,'baseline_rank':b.index(target)+1 if target in b else None,
                        'contextual_rank':a.index(target)+1 if target in a else None}
                        for key,target,b,a in zip(plan['case_ids'],targets,before,after)]}


def summarize_final(work, plan, new, baseline, actual):
    result = {**final_metrics(plan,new,baseline,actual),
        'evaluation_plan_sha256':digest(plan),'scores_sha256':sha(work/'scores.json.gz'),
        'responses_sha256':sha(work/'rerank-responses.json')}
    atomic_json(work/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k != 'case_results'},ensure_ascii=False),flush=True)


def export_artifacts(work, destination):
    """Publish successful evidence only; no credentials, vectors or rejected raw generations."""
    if destination.exists():
        raise ValueError('Use a fresh public artifact directory')
    destination.mkdir(parents=True)
    for name in ('plan.json','evaluation-plan.json','scores.json.gz','result.json','stage-results.json'):
        (destination/name).write_bytes((work/name).read_bytes())
    for name in ('prefixes.json','rerank-responses.json'):
        (destination/(name+'.gz')).write_bytes(gzip.compress((work/name).read_bytes(),mtime=0))


def replay_published(destination):
    """Verify public source/response bindings without credentials, vectors, or remote clients."""
    plan = read_json(destination/'evaluation-plan.json')
    if (plan['model'] != MODEL or plan['dimensions'] != DIMENSIONS or plan['fusion'] != BASELINE
            or plan['rerank_model'] != 'Qwen/Qwen3-Reranker-4B'
            or plan['rerank_instruction'] != ENGLISH_INSTRUCTION or plan['rerank_candidates'] != 40
            or plan['rerank_text'] != 'original_content'):
        raise ValueError('Published experiment configuration changed')
    generation = read_json(destination/'plan.json')
    raw = gzip.decompress((destination/'prefixes.json.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != plan['prefixes_sha256']:
        raise ValueError('Published prefixes changed')
    corpus = Corpus(CORPUS_DIR)
    sources = ordered_records(corpus)
    if (plan['generation_plan_sha256'] != digest(generation)
            or generation['source_fingerprint'] != digest([source_payload(r) for r in sources])
            or plan['corpus_fingerprint'] != corpus.fingerprint):
        raise ValueError('Published generation/source identity mismatch')
    saved = json.loads(raw)
    if saved['identity']['plan_sha256'] != digest(generation):
        raise ValueError('Published prefix plan mismatch')
    prefixes = {r['id']:r for r in saved['records']}
    if len(prefixes) != len(saved['records']) or set(prefixes) != {r.id for r in sources}:
        raise ValueError('Incomplete/foreign prefixes')
    texts = []
    for record in corpus.records:
        row = prefixes[record.id]
        source = source_payload(record)
        if row['source_sha256'] != digest(source):
            raise ValueError('Published prefix source mismatch')
        if row.get('rejected_reason'):
            if row['context'] or row['anchors']:
                raise ValueError('Rejected prefix must be empty')
        else:
            expected = grounded_prefix(source,{k:row[k] for k in ('id','terms','aspects')})
            if any(row[k] != expected[k] for k in expected):
                raise ValueError('Published prefix contains unsupported text')
        base = MedicalKnowledgeBase.retrieval_text(record)
        texts.append((record.id,row['context']+'\n'+base if row['context'] else base))
    if digest([(key,text_hash(text)) for key,text in texts]) != plan['document_fingerprint']:
        raise ValueError('Published index text fingerprint mismatch')
    new = read_json(destination/'scores.json.gz')
    validate_cache(new)
    if (new['metadata']['document_fingerprint'] != plan['document_fingerprint']
            or new['metadata']['embedding_model'] != MODEL or new['metadata']['embedding_dimensions'] != DIMENSIONS):
        raise ValueError('Published retrieval representation mismatch')
    response_raw = gzip.decompress((destination/'rerank-responses.json.gz').read_bytes())
    responses = json.loads(response_raw)
    if (responses['identity']['plan_sha256'] != digest(plan)
            or responses['identity']['scores_sha256'] != sha(destination/'scores.json.gz')
            or responses['identity']['model'] != plan['rerank_model']
            or responses['identity']['instruction'] != plan['rerank_instruction']
            or sha(BASELINE_RESPONSES) != plan['baseline_responses_sha256']):
        raise ValueError('Published response/score binding mismatch')
    actual = {r['case_id']:r for r in responses['completed']}
    if len(actual) != len(responses['completed']):
        raise ValueError('Duplicate responses')
    baseline = {r['case_id']:r for r in read_json(BASELINE_RESPONSES)['completed'] if r['arm']=='english_4b40'}
    result = read_json(destination/'result.json')
    computed = final_metrics(plan,new,baseline,actual)
    if any(result[k] != value for k,value in computed.items()):
        raise ValueError('Published metrics disagree with actual responses')
    if hashlib.sha256(response_raw).hexdigest() != result['responses_sha256']:
        raise ValueError('Published response bytes changed')
    print(json.dumps({'verified_records':len(prefixes),'verified_rerank_responses':len(actual),
        'external_requests':0,'baseline':computed['baseline'],'contextual':computed['contextual'],
        'paired':computed['paired']},ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','embed','score','rerank','replay','export','replay-published'))
    parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--artifact-dir',type=Path)
    parser.add_argument('--config-file',type=Path)
    parser.add_argument('--allow-remote',action='store_true')
    args = parser.parse_args()
    if args.mode == 'replay-published':
        if not args.artifact_dir:
            parser.error('Public artifact directory required')
        replay_published(args.artifact_dir)
        return
    if not args.work_dir:
        parser.error('Independent experiment work directory required')
    work = work_directory(args.work_dir)
    plan,corpus,cases,texts,queries,old,baseline = inputs(work)
    if args.mode in {'embed','rerank'}:
        if not args.allow_remote or not args.config_file:
            parser.error('Explicit remote flag and private configuration selection required')
        load_config(args.config_file)
    if args.mode == 'embed':
        embed(work,plan,texts)
    elif args.mode == 'score':
        new = build_scores(work,plan,corpus,cases,texts,queries,old)
        summarize_stages(work,plan,old,new)
    elif args.mode == 'export':
        if not args.artifact_dir:
            parser.error('New public artifact directory required')
        export_artifacts(work,args.artifact_dir)
        replay_published(args.artifact_dir)
    elif args.mode in {'rerank','replay'}:
        new = read_json(work/'scores.json.gz')
        if new['metadata']['document_fingerprint'] != plan['document_fingerprint']:
            raise ValueError('New retrieval cache identity mismatch')
        actual = rerank(work,plan,cases,corpus,new,allow_remote=args.mode=='rerank')
        summarize_final(work,plan,new,baseline,actual)
    else:
        print(json.dumps({k:v for k,v in plan.items() if k not in {'case_ids','rejected_prefixes_original_fallback'}},ensure_ascii=False))


if __name__ == '__main__':
    main()
