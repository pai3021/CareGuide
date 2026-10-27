"""Read-only inventory before retiring Chroma user-profile projections.

Print counts only. Does not copy unverified legacy prose into active preferences,
delete collections, create SQLite files, or expose user IDs/profile contents.
"""
import argparse
import json
from pathlib import Path
import sqlite3


def audit(path, collection):
    canonical = {}
    if Path(path).exists():
        with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True) as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='durable_profiles'").fetchone()
            if exists:
                canonical = {actor: json.loads(payload) for actor, payload in db.execute(
                    'SELECT actor,payload FROM durable_profiles')}
    report = {'sqlite_profiles': len(canonical), 'chroma_profiles': 0,
              'chroma_only': 0, 'different_payload': 0, 'unverified_legacy': 0}
    offset = 0
    while True:
        batch = collection.get(limit=100, offset=offset, include=['documents', 'metadatas'])
        identities = batch.get('ids') or []
        if not identities:
            break
        for document, metadata in zip(batch.get('documents') or [], batch.get('metadatas') or []):
            payload = json.loads(document)
            actor = (metadata or {}).get('user_id')
            report['chroma_profiles'] += 1
            report['chroma_only'] += int(actor not in canonical)
            report['different_payload'] += int(actor in canonical and payload != canonical[actor])
            report['unverified_legacy'] += int(payload.get('schema_version') != 2)
        offset += len(identities)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-db', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--prefix', default='careguide')
    args = parser.parse_args()
    import chromadb
    client = chromadb.HttpClient(host=args.host, port=args.port,
                                settings=chromadb.Settings(anonymized_telemetry=False))
    names = [item.name if hasattr(item, 'name') else str(item) for item in client.list_collections()]
    name = args.prefix+'_user_profile'
    if name not in names:
        print(json.dumps({'status': 'no_legacy_collection', 'prefix': args.prefix}))
    else:
        print(json.dumps(audit(args.profile_db, client.get_collection(name)), ensure_ascii=False, indent=2))
