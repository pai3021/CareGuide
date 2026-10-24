"""Explicit no-fee vector reuse; needs a trusted original-corpus hash export."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from careguide.corpus import Corpus
from careguide.retrieval.index_reuse import reuse_verified_index
from careguide.runtime import chroma_client


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--corpus', default='data/careguide_kuake_rag_v1')
    parser.add_argument('--index-path', help='Explicit existing local Chroma directory')
    args = parser.parse_args()
    if args.index_path:
        import chromadb
        client = chromadb.PersistentClient(path=args.index_path,
            settings=chromadb.Settings(anonymized_telemetry=False))
    else:
        client = chroma_client()
    result = reuse_verified_index(client, Corpus(args.corpus), args.source,
        json.loads(Path(args.binding).read_text(encoding='utf-8')),
        model=os.environ.get('CAREGUIDE_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B'),
        dimensions=int(os.environ.get('CAREGUIDE_EMBEDDING_DIMENSIONS', '1024')))
    print(json.dumps(result, ensure_ascii=False, indent=2))
