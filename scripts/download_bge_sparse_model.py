"""Download pinned public BGE-M3 weights and verify upstream file identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
FILES = {'config.json', 'pytorch_model.bin', 'sparse_linear.pt', 'colbert_linear.pt', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model'}


def file_hash(path, algorithm='sha256', *, git_blob=False):
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mirror', choices=('huggingface', 'modelscope'), default='huggingface')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    import requests
    metadata = requests.get(f'https://huggingface.co/api/models/BAAI/bge-m3/revision/{REVISION}?blobs=true',
                            timeout=60).json()
    if metadata['sha'] != REVISION:
        raise ValueError('Unexpected upstream model revision')
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for item in metadata['siblings']:
        name = item['rfilename']
        if name not in FILES:
            continue
        path = args.output/name
        part = path.with_suffix(path.suffix+'.part')
        url = (f'https://modelscope.cn/models/BAAI/bge-m3/resolve/master/{name}'
               if args.mirror == 'modelscope' else
               f'https://huggingface.co/BAAI/bge-m3/resolve/{REVISION}/{name}?download=true')
        if not path.exists() and item['size'] > 128*1024*1024:
            parts = args.output/('.download-'+name)
            parts.mkdir(exist_ok=True)
            chunk_size = 16*1024*1024
            def fetch(start):
                end = min(start+chunk_size, item['size'])-1
                chunk_path = parts/str(start)
                if chunk_path.exists() and chunk_path.stat().st_size == end-start+1:
                    return chunk_path
                for attempt in range(3):
                    try:
                        with requests.get(url+('&' if '?' in url else '?')+f'chunk={start}',
                                          headers={'Range': f'bytes={start}-{end}'},
                                          stream=True, timeout=(30, 90)) as response:
                            if response.status_code != 206 or not response.headers.get(
                                    'Content-Range', '').startswith(f'bytes {start}-{end}/'):
                                raise ValueError('Server did not honor bounded range')
                            temporary = chunk_path.with_suffix('.downloading')
                            with temporary.open('wb') as handle:
                                for chunk in response.iter_content(1024*1024):
                                    handle.write(chunk)
                            if temporary.stat().st_size != end-start+1:
                                raise ValueError('Incomplete download chunk')
                            temporary.replace(chunk_path)
                        return chunk_path
                    except Exception:
                        if attempt == 2:
                            raise RuntimeError(f'Model range download failed at byte {start}') from None
                        time.sleep(2)
            starts = list(range(0, item['size'], chunk_size))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(fetch, start) for start in starts]
                for completed, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if completed % 8 == 0 or completed == len(starts):
                        print(f'{name}: downloaded {completed}/{len(starts)} verified-size ranges', flush=True)
            with part.open('wb') as handle:
                for start in starts:
                    with (parts/str(start)).open('rb') as chunk_handle:
                        for chunk in iter(lambda: chunk_handle.read(8*1024*1024), b''):
                            handle.write(chunk)
            part.replace(path)
        if not path.exists():
            for attempt in range(3):
                offset = part.stat().st_size if part.exists() else 0
                try:
                    with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                                      stream=True, timeout=(30, 90)) as response:
                        response.raise_for_status()
                        append = offset > 0 and response.status_code == 206
                        count = offset if append else 0
                        progress = count//(64*1024*1024)
                        with part.open('ab' if append else 'wb') as handle:
                            for chunk in response.iter_content(1024*1024):
                                handle.write(chunk)
                                count += len(chunk)
                                if count//(64*1024*1024) > progress:
                                    progress = count//(64*1024*1024)
                                    print(f'{name}: {count}/{item["size"]} bytes', flush=True)
                    if count != item['size']:
                        raise ValueError('Downloaded model file size mismatch')
                    part.replace(path)
                    break
                except Exception as exc:
                    print(f'{name}: download attempt {attempt+1}: {type(exc).__name__}', flush=True)
                    if attempt == 2:
                        raise RuntimeError('Model download could not complete') from None
                    time.sleep(2)
        expected = (item.get('lfs') or {}).get('sha256')
        actual = file_hash(path)
        if path.stat().st_size != item['size'] or (expected and actual != expected):
            raise ValueError('Model weight checksum/size mismatch')
        if not expected and file_hash(path, 'sha1', git_blob=True) != item['blobId']:
            raise ValueError('Model configuration Git blob mismatch')
        records[name] = {'size': path.stat().st_size, 'sha256': actual}
        print(f'verified {name}', flush=True)
    if set(records) != FILES:
        raise ValueError('Incomplete model download')
    (args.output/'careguide_model_lock.json').write_text(json.dumps(
        {'model': 'BAAI/bge-m3', 'revision': REVISION, 'download_mirror': args.mirror,
         'files': records}, indent=2)+'\n', encoding='utf-8')
    print('Pinned BGE-M3 sparse model complete', flush=True)


if __name__ == '__main__':
    main()
