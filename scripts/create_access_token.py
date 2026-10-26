"""Provision an opaque token. Only its SHA256 digest is stored; stdout shows it once."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--role', choices=['user', 'admin'], default='user')
    args = parser.parse_args()
    if not 1 <= len(args.user) <= 100:
        parser.error('User identity must contain 1 to 100 characters')
    data = json.loads(args.file.read_text(encoding='utf-8')) if args.file.exists() else {'users': []}
    if any(row['user_id'] == args.user for row in data['users']):
        parser.error('User already exists; remove its old digest explicitly to rotate access')
    token = secrets.token_urlsafe(32)
    data['users'].append({'user_id': args.user, 'role': args.role, 'token_sha256': hashlib.sha256(token.encode()).hexdigest()})
    args.file.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    print(token)
