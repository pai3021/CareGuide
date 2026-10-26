import MarkdownIt from 'markdown-it'

const md = new MarkdownIt({ html: false, linkify: false, breaks: true })
md.disable('image')
md.validateLink = url => /^(https?:\/\/|#[A-Za-z0-9_-]+$)/i.test(url)
const linkOpen = md.renderer.rules.link_open || ((tokens, index, options, env, self) => self.renderToken(tokens, index, options))
md.renderer.rules.link_open = (tokens, index, options, env, self) => {
  tokens[index].attrSet('target', '_blank')
  tokens[index].attrSet('rel', 'noopener noreferrer')
  return linkOpen(tokens, index, options, env, self)
}
md.inline.ruler.before('link', 'source_reference', (state, silent) => {
  const match = /^\[(S\d+)\]/.exec(state.src.slice(state.pos))
  if (!match || !state.env.sourceIds?.has(match[1])) return false
  if (!silent) {
    const token = state.push('source_reference', '', 0)
    token.content = match[1]
  }
  state.pos += match[0].length
  return true
})
md.renderer.rules.source_reference = (tokens, index) => {
  const id = tokens[index].content // generated only by the strict S<number> rule
  return `<a href="#${id}" class="source-reference" data-source-id="${id}">[${id}]</a>`
}

export function renderMarkdown(text, sources = []) {
  return md.render(String(text || ''), { sourceIds: new Set(sources.map(source => source.id).filter(id => /^S\d+$/.test(id))) })
}

export function safeSourceUrl(value) {
  return /^https?:\/\//i.test(String(value || '')) ? value : undefined
}
