const tokens = new Map()
const key = base => `careguide.access:${String(base).replace(/\/+$/, '')}`
export function accessToken(base) {
  try { return tokens.get(key(base)) || sessionStorage.getItem(key(base)) || '' } catch { return tokens.get(key(base)) || '' }
}
export function setAccessToken(base, token) {
  const name = key(base), value = token.trim()
  tokens.set(name, value)
  try { if (value) sessionStorage.setItem(name, value); else sessionStorage.removeItem(name) } catch {}
}
export function authHeaders(base) {
  const token = accessToken(base)
  return token ? { Authorization: `Bearer ${token}` } : {}
}
