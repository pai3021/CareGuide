"""Validate cached source identities; optionally check official pages without renewal."""
import argparse
import hashlib
import json
from pathlib import Path

from careguide.corpus import Corpus
from scripts.inspect_source_pages import extract

ROOT = Path(__file__).resolve().parents[1]
FILES = {'nmpa-otc-labels': 'nmpa-otc.pdf', 'pkuih-appointment': 'pkuih-appointment.html',
         'pkufh-daxing-appointment': 'pkufh-daxing-appointment.html',
         'pkuih-neurology': 'pkuih-neurology.html',
         'pkuih-gastroenterology': 'pkuih-gastroenterology.html', 'nhs-headaches': 'nhs-headaches.html'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    corpus = Corpus()
    sources = json.loads((ROOT/'data/medical_v1/sources.json').read_text(encoding='utf-8'))
    results = []
    for source in sources:
        cache = ROOT/'data/source_cache'/FILES[source['id']]
        if args.online:
            import httpx
            response = httpx.get(source['url'], timeout=45, follow_redirects=True)
            response.raise_for_status()
            raw = response.content
        else:
            raw = cache.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        matches = digest == source['retrieved_sha256']
        if not args.online and not matches:
            raise ValueError(f'Cached source snapshot mismatch: {source["id"]}')
        if source['verification_method'] == 'developer_html_source_verification':
            text = extract(raw)
            if not all(anchor in text for anchor in source['evidence_anchors']):
                raise ValueError(f'Source evidence anchors missing: {source["id"]}')
        elif not raw.startswith(b'%PDF'):
            raise ValueError('Regulator document is not a PDF')
        results.append({'source_id': source['id'], 'snapshot_sha256_matches': matches,
                        'manual_review_required': args.online and not matches})
    print(json.dumps({'corpus_fingerprint': corpus.fingerprint, 'sources': results,
                      'validity_renewed': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
