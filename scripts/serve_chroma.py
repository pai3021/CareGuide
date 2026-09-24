"""Run an independent native Chroma server for CareGuide memory validation."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=Path('data/runtime/chroma-server'))
    parser.add_argument('--port', type=int, default=8002)
    args = parser.parse_args()
    import chromadb
    from chromadb.server.fastapi import FastAPI
    import uvicorn
    server = FastAPI(chromadb.Settings(is_persistent=True,
        persist_directory=str(args.path.resolve()), anonymized_telemetry=False))
    uvicorn.run(server.app(), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
