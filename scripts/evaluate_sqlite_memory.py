"""Fixed synthetic Memory probes; no remote model calls or hospital actions.

Reports FTS retrieval behavior separately from scripted source validation. The
Redis/Chroma comparison disables semantic retrieval to isolate storage and
compaction; it cannot establish parity with real embedding retrieval.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from careguide.archived_memory import ArchivedMemory
from careguide.sqlite_memory import SQLiteMemoryBackend
from scripts.benchmark_context_memory import measure


# Each query has a manually specified relevant statement, independent of ranking.
CASES = [
    ('我习惯使用简短回答', '简短回答', '别解释得太啰嗦'),
    ('我平时工作需要轮夜班', '轮夜班', '白天我通常要补觉'),
    ('我长期从南京出发看病', '南京出发', '考虑我跨城就医的路程'),
    ('我有一个需要照顾的父亲', '照顾父亲', '安排时间时考虑家里的老人'),
    ('我习惯用中文交流', '中文交流', '不要使用英文解释'),
    ('我平时坐轮椅出行', '轮椅出行', '要选择无障碍通道'),
    ('我平时使用公共交通就诊', '公共交通', '地铁公交能不能到'),
    ('我平时只能周末去医院', '周末去医院', '工作日我没法去'),
]


class ScriptedExtraction:
    """Return the provided fixture verbatim, not a simulated quality score."""
    def __init__(self):
        self.calls = 0
        self.invalid = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls += 1
        sources = json.loads(kwargs['messages'][0]['content'])['sources']
        identity, source = next(iter(sources.items()))
        quote = source['clauses'][0]
        content = 'not present in the source' if self.invalid else quote
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
            {'memories': [dict(content=content, source_quote=quote, source_message_id=identity)]},
            ensure_ascii=False)}}]}


async def complete(memory, actor, text, request):
    memory.begin_turn(actor, 'fixture', request, text)
    return await memory.complete_turn(actor, 'fixture', request, text, {'response': '合成测试回复。'})


async def evaluate(directory):
    model = ScriptedExtraction()
    path = directory/'retrieval.sqlite3'
    backend = SQLiteMemoryBackend(path, client=model, model='scripted-extraction')
    memory = ArchivedMemory(backend, path)
    ids = {}
    for index, (content, _, _) in enumerate(CASES):
        outcome = await complete(memory, 'probe-user', content, str(index))
        if outcome['status'] != 'updated':
            raise AssertionError(outcome)
        saved = backend.manager._profile_store.memories.search('probe-user', content, limit=20)
        ids[content] = next(row['id'] for row in saved if row['content'] == content)
    probes = []
    for content, lexical, paraphrase in CASES:
        for kind, query in [('lexical', lexical), ('paraphrase', paraphrase)]:
            hits = await backend.manager._get_long_term_memories('probe-user', query)
            probes.append(dict(kind=kind, query=query, expected_content=content,
                hit_at_5=ids[content] in {row['id'] for row in hits},
                retrieved_contents=[row['content'] for row in hits]))
    retrieval = {kind: dict(queries=8, recall_at_5=sum(
        row['hit_at_5'] for row in probes if row['kind'] == kind)/8)
        for kind in ('lexical', 'paraphrase')}
    source_checks = []
    for index, (text, expected) in enumerate([
        ('今天我下午去医院', 'skipped'),
        ('如果我长期住在北京', 'skipped'),
        ('我习惯使用简短回答', 'updated'),
    ]):
        outcome = await complete(memory, 'gate-user', text, 'gate-'+str(index))
        source_checks.append(dict(expected=expected, actual=outcome['status']))
    model.invalid = True
    invalid = await complete(memory, 'invalid-user', '我习惯使用简短回答', 'invalid')
    source_checks.append(dict(expected='unavailable', actual=invalid['status']))
    if backend.manager._profile_store.memories.snapshot('invalid-user')[1]:
        raise AssertionError('Unsupported scripted content was persisted')
    await memory.close()
    return dict(fts5_retrieval=retrieval, retrieval_probes=probes,
        scripted_source_checks=source_checks,
        scripted_source_checks_passed=sum(row['expected'] == row['actual'] for row in source_checks),
        real_model_extraction_accuracy=None, real_embedding_recall_comparison=None)


async def run(output):
    Path('tmp').mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='sqlite-memory-eval-', dir='tmp') as temporary:
        directory = Path(temporary)
        report = dict(protocol='careguide-sqlite-memory-v1',
            scope='Synthetic storage/control-flow comparison and 16 hand-authored FTS5 probes; not a real-model quality benchmark',
            token_method='UTF-8 heuristic, not provider usage',
            legacy_semantic_retrieval='disabled to isolate compaction and persistence',
            models='scripted fixtures', hospital_actions_executed=False,
            context_comparison=[await measure(mode, directory) for mode in ('enhanced', 'sqlite_only')],
            **await evaluate(directory))
        report['source_sha256'] = {str(path).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path('careguide')/name for name in ('sqlite_memory.py', 'long_term_memory.py',
                'memory_types.py', 'archived_memory.py', 'session_store.py', 'context_memory.py',
                'content_memory.py', 'profile_memory.py', 'conversation_memory.py')]
            + [Path(__file__), Path('scripts/benchmark_context_memory.py')]}
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(json.dumps({key: report[key] for key in (
            'context_comparison', 'fts5_retrieval', 'scripted_source_checks_passed')},
            ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    asyncio.run(run(parser.parse_args().output))
