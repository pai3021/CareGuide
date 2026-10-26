<template>
  <section class="answer-body" ref="root">
    <div class="answer-markdown" v-html="html" @click="followSource" />
    <details v-if="sources.length" class="answer-sources" :open="Boolean(selected)">
      <summary>资料来源 · {{ sources.length }} 项</summary>
      <article v-for="source in sources" :key="source.id" :data-source="source.id" :class="{ selected: source.id === selected }" tabindex="-1">
        <strong>[{{ source.id }}] {{ source.title || source.task_id }}</strong>
        <p v-if="source.excerpt">{{ source.excerpt }}</p>
        <pre v-if="source.items?.length">{{ JSON.stringify(source.items, null, 2) }}</pre>
        <small v-if="source.review_status === 'open_dataset_unreviewed'">开放研究资料，未经临床审核</small>
        <a v-if="safeSourceUrl(source.url)" :href="safeSourceUrl(source.url)" target="_blank" rel="noopener noreferrer">查看原始来源</a>
      </article>
    </details>
  </section>
</template>

<script setup>
import { computed, nextTick, ref } from 'vue'
import { renderMarkdown, safeSourceUrl } from '../lib/markdown'
const props = defineProps({ content: { type: String, default: '' }, sources: { type: Array, default: () => [] } })
const root = ref(null), selected = ref('')
const html = computed(() => renderMarkdown(props.content, props.sources))
async function followSource(event) {
  const id = event.target.closest('[data-source-id]')?.dataset.sourceId
  if (!id || !props.sources.some(source => source.id === id)) return
  event.preventDefault()
  selected.value = id
  await nextTick()
  const card = root.value.querySelector(`[data-source="${id}"]`)
  card?.focus({ preventScroll: true })
  card?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
}
</script>

<style scoped>
.answer-markdown { overflow-wrap: anywhere; line-height: 1.7; }
.answer-markdown :deep(p) { white-space: normal; }
.answer-markdown :deep(pre), pre { overflow: auto; white-space: pre-wrap; font-size: .85em; }
.answer-markdown :deep(table) { border-collapse: collapse; display: block; overflow-x: auto; }
.answer-markdown :deep(th), .answer-markdown :deep(td) { border: 1px solid #8885; padding: .4em .7em; }
.answer-markdown :deep(blockquote) { border-left: 3px solid #8887; padding-left: 1em; margin-left: 0; }
.answer-markdown :deep(a), a { color: #17796e; }
.answer-sources { margin-top: 1em; font-size: .9em; }
article { padding: .8em; border: 1px solid #8884; border-radius: .5em; margin-top: .6em; }
article.selected { outline: 2px solid #378a80; }
article p { white-space: pre-wrap; }
small { display: block; }
</style>
