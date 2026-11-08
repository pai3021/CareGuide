"""Grounding, source-only payload and provider boundaries for enrichment."""
from types import SimpleNamespace
import asyncio
import json

import pytest

from scripts.contextualize_rag_corpus import grounded_prefix, normalize_batch, require_mimo_endpoint, source_payload, validate_prefixes


def example():
    source = {'id':'a','content':'检查前需要空腹，糖尿病患者请提前咨询医生。'}
    row = {'id':'a','context':'本段介绍检查前空腹要求，以及糖尿病患者咨询医生的注意事项。',
           'anchors':['检查前需要空腹','糖尿病患者请提前咨询医生']}
    return source, row


def test_enrichment_evidence_must_come_from_the_same_record():
    source, row = example()
    assert validate_prefixes([source], {'records':[row]}) == [row]
    with pytest.raises(ValueError, match='source substrings'):
        validate_prefixes([source], {'records':[{**row,'anchors':['检查前需要空腹','阿莫西林']}]})
    with pytest.raises(ValueError, match='identity'):
        validate_prefixes([source], {'records':[{**row,'id':'other'}]})


def test_generation_payload_does_not_leak_query_labels_or_generic_titles():
    source = SimpleNamespace(id='doc', content='公开原文', representation='open_dataset_qa',
        title='KUAKE-IR 开放医学段落 doc', keywords=['KUAKE-IR'], query='do not send', gold_record_id='gold')
    assert source_payload(source) == {'id':'doc','content':'公开原文'}


@pytest.mark.parametrize('url',['https://other.invalid/v1','http://api.xiaomimimo.com/v1',
    'https://token-plan-cn.xiaomimimo.com/v1?secret=1','https://user@api.xiaomimimo.com/v1'])
def test_enrichment_cannot_send_to_a_different_endpoint(url):
    with pytest.raises(ValueError, match='MiMo endpoint'):
        require_mimo_endpoint(url)


def test_prefix_validation_rejects_incomplete_duplicate_and_oversized_results():
    source, row = example()
    for rows in ([], [row,row], [{**row,'context':'字'*141}], [{**row,'anchors':['检查','检查']}],
                 [{**row,'query':'a query must never enter the saved prefix'}]):
        with pytest.raises(ValueError):
            validate_prefixes([source], {'records':rows})


def test_grounded_prefix_cannot_invent_subjects_or_medical_assertions():
    source = {'id':'a','content':'有先兆流产的可能，建议到医院进一步检查。'}
    prefix = grounded_prefix(source,{'id':'a','terms':['先兆流产'],'aspects':['检查与诊断']})
    assert prefix['context'] == '本段涉及先兆流产，讨论检查与诊断。'
    for row in ({'id':'a','terms':['阴道出血'],'aspects':['检查与诊断']},
                {'id':'a','terms':['先兆流产'],'aspects':['已确诊']},
                {'id':'a','terms':['医院'],'aspects':['就医流程']}):
        with pytest.raises(ValueError):
            grounded_prefix(source,row)


def test_batch_reordering_and_wrong_ids_never_move_text_to_a_different_source():
    sources=[{'id':'a','content':'头晕可以进一步检查。'},{'id':'b','content':'腹泻注意日常饮食。'}]
    a={'id':'a','terms':['头晕'],'aspects':['检查与诊断']}
    b={'id':'b','terms':['腹泻'],'aspects':['饮食建议']}
    rows=normalize_batch(sources,{'records':[b,a]})
    assert rows[0]['terms']==['头晕'] and rows[1]['terms']==['腹泻']
    rows=normalize_batch(sources,{'records':[a,{**b,'id':'foreign'}]})
    assert rows[0]['context'] and rows[1]['context']==''
    rows=normalize_batch(sources,{'records':[a,a,b]})
    assert rows[0]['context']=='' and rows[1]['context']


@pytest.mark.parametrize('provider_filtered',[False,True])
def test_generation_checkpoint_reuses_calls_and_falls_back_for_unsupported_subjects(tmp_path,monkeypatch,provider_filtered):
    from scripts.contextualize_rag_corpus import generate, MODEL
    import openai
    calls = []
    class Client:
        def __init__(self,**kwargs):
            assert kwargs['max_retries'] == 0
            self.chat = SimpleNamespace(completions=self)
        async def create(self,**kwargs):
            payload = json.loads(kwargs['messages'][1]['content'])['records']
            calls.append(payload)
            if provider_filtered and payload[0]['id']=='0':
                return SimpleNamespace(usage=None,choices=[SimpleNamespace(finish_reason='content_filter',
                    message=SimpleNamespace(content='Provider filter'))])
            result = [{'id':r['id'],'terms':['阴道出血' if r['id']=='0' else '头晕'],
                       'aspects':['检查与诊断']} for r in payload]
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100,completion_tokens=50),
                choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps({'records':result})))])
        async def close(self): pass
    monkeypatch.setattr(openai,'AsyncOpenAI',Client)
    monkeypatch.setenv('CAREGUIDE_LLM_BASE_URL','https://token-plan-cn.xiaomimimo.com/v1')
    monkeypatch.setenv('CAREGUIDE_LLM_MODEL',MODEL)
    records = [SimpleNamespace(id=str(i),content='头晕可以进一步检查，先注意休息。',representation='open_dataset_qa')
               for i in range(201)]
    plan = {'maximum_llm_requests':12,'max_output_tokens_per_request':4096}
    asyncio.run(generate(tmp_path,plan,records,limit=200,concurrency=2))
    assert len(calls) == 7
    saved = json.loads((tmp_path/'prefixes.json').read_text(encoding='utf-8'))
    rejected = next(r for r in saved['records'] if r['id']=='0')
    assert rejected['context'] == '' and rejected['rejected_reason']
    if provider_filtered:
        assert sum('Provider content filter' in r.get('rejected_reason','') for r in saved['records']) == 32
    asyncio.run(generate(tmp_path,plan,records,limit=200,concurrency=2))
    assert len(calls) == 7
    ledger = json.loads((tmp_path/'llm-requests.json').read_text(encoding='utf-8'))
    assert len(ledger['attempts']) == 7
    with pytest.raises(ValueError,match='Checkpoint/ledger'):
        asyncio.run(generate(tmp_path,{**plan,'maximum_llm_requests':13},records,limit=200,concurrency=2))
    assert len(calls) == 7


def test_final_report_refuses_reranker_outputs_from_a_different_pool(tmp_path):
    from scripts.evaluate_contextual_retrieval import summarize_final
    row={'id':'q','gold_record_id':'a','streams':{'dense':[('a',0.9),('b',0.5)],'bm25':[('b',1.0)]}}
    actual={'q':{'candidate_ids':['other'],'ids':['other']}}
    with pytest.raises(ValueError,match='Candidate/response'):
        summarize_final(tmp_path,{'case_ids':['q']},{'rows':[row]}, {}, actual)


def test_parallel_document_embeddings_resume_without_extra_calls_and_reject_changed_text(tmp_path,monkeypatch):
    import sqlite3
    from scripts import evaluate_contextual_retrieval as trial
    calls = []
    class Provider:
        def __init__(self,**kwargs):
            assert kwargs['max_retries'] == 0
            self.observe = kwargs['response_observer']
        def embed_documents(self,texts):
            assert (tmp_path/'embedding-requests.json').is_file()
            calls.append(tuple(texts))
            self.observe(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=len(texts)*10)))
            return [[1.0]*1024 for _ in texts]
        def close(self): pass
    monkeypatch.setattr(trial,'OpenAICompatibleEmbeddingFunction',Provider)
    monkeypatch.setenv('CAREGUIDE_EMBEDDING_BASE_URL','https://api.siliconflow.cn/v1')
    plan={'maximum_embedding_requests':4,'query_instruction':'fixed instruction'}
    texts=[(str(i),'source '+str(i)) for i in range(70)]
    trial.embed(tmp_path,plan,texts)
    assert len(calls) == 3
    trial.embed(tmp_path,plan,texts)
    assert len(calls) == 3
    with sqlite3.connect(tmp_path/'vectors.sqlite3') as db:
        assert db.execute('select count(*) from vectors').fetchone()[0] == 70
    ledger=json.loads((tmp_path/'embedding-requests.json').read_text(encoding='utf-8'))
    assert len(ledger['attempts']) == 3
    assert sum(row['input_tokens'] for row in ledger['attempts']) == 700
    with pytest.raises(ValueError,match='Changed input'):
        trial.embed(tmp_path,plan,[(key,text+'changed') for key,text in texts])
    assert len(calls) == 3
