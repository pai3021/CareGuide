"""Validate first; --apply explicitly sends public corpus notes to embedding API."""
import argparse

from careguide.corpus import Corpus
from careguide.retrieval.knowledge_base import COLLECTION_NAME, index_collection
from careguide.runtime import chroma_client, semantic_embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--versioned', action='store_true')
    parser.add_argument('--config-file')
    parser.add_argument('--corpus-dir', help='Explicit corpus directory; defaults to the frozen medical_v1 corpus')
    args = parser.parse_args()
    from scripts.private_config import load_config
    load_config(args.config_file)
    corpus = Corpus(args.corpus_dir)
    from careguide.business_store import BusinessStore
    from careguide.knowledge_workflow import KnowledgeWorkflow
    from careguide.runtime import PROJECT_ROOT
    import os
    corpus = KnowledgeWorkflow(BusinessStore(os.environ.get('CAREGUIDE_BUSINESS_DB',
        str(PROJECT_ROOT/'data'/'runtime'/'medical.sqlite3')))).extend(corpus)
    from careguide.retrieval.knowledge_base import versioned_collection_name
    target = versioned_collection_name(corpus) if args.versioned else COLLECTION_NAME
    print(f'Verified {len(corpus.records)} notes; target collection: {target}')
    if not args.apply:
        print('Dry run: no collection writes, no external calls.')
        return
    embedding = semantic_embedding()
    collection = index_collection(chroma_client(), corpus, embedding, versioned=args.versioned)
    print(f'Indexed {collection.count()} notes into the independent medical collection.')


if __name__ == '__main__':
    main()
