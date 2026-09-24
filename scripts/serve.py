"""Launch CareGuide with explicitly selected private configuration."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--memory-mode', choices=['session', 'redis_chroma'])
    parser.add_argument('--retrieval-mode', choices=['bm25', 'dense', 'hybrid'])
    parser.add_argument('--persist-traces', action='store_true')
    args = parser.parse_args()
    from scripts.private_config import load_config
    load_config(args.config_file)
    if args.memory_mode:
        os.environ['CAREGUIDE_MEMORY_MODE'] = args.memory_mode
    if args.retrieval_mode:
        os.environ['CAREGUIDE_RETRIEVAL_MODE'] = args.retrieval_mode
    if args.persist_traces:
        os.environ['CAREGUIDE_PERSIST_TRACES'] = 'true'
    import uvicorn
    uvicorn.run('careguide.api:app', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
