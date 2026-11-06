"""One explicitly requested SiliconFlow BGE capability probe; no secret output."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.private_config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Probe already recorded; refusing an implicit repeat')
    load_config(args.config_file)
    endpoint = os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL', '').rstrip('/')
    if endpoint != 'https://api.siliconflow.cn/v1':
        raise ValueError('Probe is restricted to the configured SiliconFlow endpoint')
    key = os.environ.get('CAREGUIDE_EMBEDDING_API_KEY', '')
    if not key:
        raise ValueError('Embedding key is unconfigured')
    report = {'model': 'BAAI/bge-m3', 'endpoint': endpoint+'/embeddings',
              'attempted_requests': 1, 'sdk_retries': 0,
              'input_kind': 'one synthetic public medical retrieval query',
              'requested_sparse': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    import httpx
    try:
        response = httpx.post(endpoint+'/embeddings', headers={'Authorization': 'Bearer '+key},
                              json={'model': 'BAAI/bge-m3', 'input': ['阿司匹林有哪些注意事项？'],
                                    'encoding_format': 'float', 'return_sparse': True},
                              timeout=60, follow_redirects=False)
        report['http_status'] = response.status_code
        body = response.json()
        report['top_level_fields'] = sorted(body) if isinstance(body, dict) else []
        data = body.get('data', []) if isinstance(body, dict) else []
        report['item_fields'] = sorted(data[0]) if data and isinstance(data[0], dict) else []
        report['dense_dimensions'] = len(data[0].get('embedding', [])) if data else 0
        report['sparse_field_present'] = any(k in {'lexical_weights', 'sparse', 'sparse_embedding',
                                                  'sparse_embeddings'}
                                             for k in report['top_level_fields']+report['item_fields'])
    except Exception as exc:
        report['error_type'] = type(exc).__name__
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
