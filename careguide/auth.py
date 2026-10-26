"""Optional opaque bearer credentials bound to server-owned actors and roles.

The provisioning file contains token SHA256 digests, never plaintext credentials.
Local demo mode is explicit in /auth/me; this is not an OIDC/SSO implementation.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import hmac
import json
import os
from pathlib import Path
import re

from fastapi import HTTPException
from pydantic import BaseModel, model_validator
from starlette.responses import JSONResponse

_principal = ContextVar('careguide_principal', default=None)


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str


def principal():
    return _principal.get()


def bound_actor(fallback):
    current = principal()
    return current.user_id if current else fallback


class ActorModel(BaseModel):
    @model_validator(mode='after')
    def bind_actor(self):
        current = principal()
        if current and hasattr(self, 'user_id'):
            if 'user_id' in self.model_fields_set and self.user_id != current.user_id:
                raise HTTPException(403, '不能使用其他用户的身份')
            self.user_id = current.user_id
        return self


@lru_cache(maxsize=8)
def _read_users(path, modified, size):
    if size > 100_000:
        raise ValueError('Credential file too large')
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['users']
    if not isinstance(rows, list) or not rows:
        raise ValueError('No credentials configured')
    seen, result = set(), []
    for row in rows:
        if (not isinstance(row.get('user_id'), str) or not 1 <= len(row['user_id']) <= 100
                or not re.fullmatch(r'[a-f0-9]{64}', row.get('token_sha256', ''))
                or row.get('role') not in {'user', 'admin'} or row['token_sha256'] in seen):
            raise ValueError('Invalid credential entry')
        seen.add(row['token_sha256'])
        result.append((row['token_sha256'], Principal(row['user_id'], row['role'])))
    return result


def authenticate(header):
    path = os.environ.get('CAREGUIDE_AUTH_FILE', '').strip()
    setting = os.environ.get('CAREGUIDE_REQUIRE_AUTH', 'false').casefold()
    if setting not in {'true', '1', 'yes', 'false', '0', 'no', ''}:
        raise HTTPException(503, '身份认证配置不可用')
    required = bool(path) or setting in {'true', '1', 'yes'}
    if not required:
        return None
    try:
        info = Path(path).stat() if path else None
        if info is None:
            raise ValueError('Missing credential file')
        users = _read_users(path, info.st_mtime_ns, info.st_size)
    except Exception:
        raise HTTPException(503, '身份认证配置不可用') from None
    scheme, _, token = (header or '').partition(' ')
    if scheme.casefold() != 'bearer' or not 32 <= len(token) <= 512:
        raise HTTPException(401, '请提供有效的访问令牌', headers={'WWW-Authenticate': 'Bearer'})
    digest = hashlib.sha256(token.encode()).hexdigest()
    for expected, identity in users:
        if hmac.compare_digest(expected, digest):
            return identity
    raise HTTPException(401, '访问令牌无效', headers={'WWW-Authenticate': 'Bearer'})


def admin_path(path, method):
    return (path.startswith('/eval/') or path in {'/monitor', '/metrics', '/skills/reload'}
            or path.startswith('/knowledge/') and path != '/knowledge/stats'
            or path == '/tools/query')


async def authentication_middleware(request, call_next):
    if request.method == 'OPTIONS' or request.url.path in {'/health', '/docs', '/openapi.json', '/redoc'}:
        return await call_next(request)
    token = None
    try:
        identity = authenticate(request.headers.get('authorization'))
        if identity:
            supplied = request.query_params.getlist('user_id')
            if any(value != identity.user_id for value in supplied):
                raise HTTPException(403, '不能访问其他用户的记录')
            if admin_path(request.url.path, request.method) and identity.role != 'admin':
                raise HTTPException(403, '此接口仅限管理员')
        token = _principal.set(identity)
        return await call_next(request)
    except HTTPException as exc:
        return JSONResponse({'detail': exc.detail}, status_code=exc.status_code, headers=exc.headers)
    finally:
        if token is not None:
            _principal.reset(token)


def owned_trace(value):
    identity = principal()
    if identity and identity.role != 'admin' and value and value.get('user_id') != identity.user_id:
        return None
    return value
