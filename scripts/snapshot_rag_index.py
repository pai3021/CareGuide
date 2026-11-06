"""Copy and hash an existing verified 0.6B index without opening Chroma writable."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from careguide.corpus import Corpus
from careguide.retrieval.knowledge_base import versioned_collection_name
from scripts.cache_rag_recall_scores import atomic_json
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR


def tree_hashes(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(handle,'sha256').hexdigest()
    return result


def inspect_index(source, corpus):
    dbpath = source/'chroma.sqlite3'
    with closing(sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        row = db.execute('select id,name,dimension from collections where name=?',
                         (versioned_collection_name(corpus),)).fetchone()
        if not row:
            raise ValueError('Required current-corpus collection missing')
        identity,name,dimension = row
        metadata = {key:next((v for v in values if v is not None),None)
                    for key,*values in db.execute('select key,str_value,int_value,float_value,bool_value '
                        'from collection_metadata where collection_id=?',(identity,))}
        ids = {value for value, in db.execute('select e.embedding_id from embeddings e '
               'join segments s on e.segment_id=s.id where s.collection=?',(identity,))}
    if (metadata.get('embedding_model') != 'Qwen/Qwen3-Embedding-0.6B'
            or dimension != 1024 or metadata.get('embedding_dimensions') != 1024
            or metadata.get('corpus_fingerprint') != corpus.fingerprint
            or metadata.get('reuse_complete') == 0
            or ids != {record.id for record in corpus.records}):
        raise ValueError('Baseline index model/corpus/completeness mismatch')
    return {'collection':name,'dimension':dimension,'record_count':len(ids),'metadata':metadata}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    source,destination=args.source.resolve(),args.destination.resolve()
    if not source.is_dir() or destination.exists() or source == destination or source in destination.parents:
        parser.error('Requires an existing source and a new separate destination')
    verified=inspect_index(source,Corpus(CORPUS_DIR))
    before=tree_hashes(source)
    shutil.copytree(source,destination)
    copied=tree_hashes(destination)
    after=tree_hashes(source)
    if before != after or before != copied:
        raise RuntimeError('Index changed while copying; snapshot is not certified')
    assert inspect_index(destination,Corpus(CORPUS_DIR)) == verified
    manifest={'source':str(source),'destination':str(destination),'verified_collection':verified,
              'files':copied,'source_unchanged':True,'external_requests':0}
    atomic_json(destination.parent/(destination.name+'.snapshot.json'),manifest)
    print(json.dumps({'verified_collection':verified,'file_count':len(copied),
                      'source_unchanged':True,'destination':str(destination)},ensure_ascii=False))


if __name__=='__main__':
    main()
