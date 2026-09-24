"""Source-scoped add/upload/review/publish workflow replacing unrestricted uploads."""
import hashlib
import json
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, model_validator
from careguide.corpus import Source, Record


class KnowledgeBundle(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sources: list[Source] = Field(min_length=1, max_length=20)
    records: list[Record] = Field(min_length=1, max_length=100)

    @model_validator(mode='after')
    def validate_links(self):
        sources = {source.id: source for source in self.sources}
        if len(sources) != len(self.sources) or len({record.id for record in self.records}) != len(self.records):
            raise ValueError('Duplicate knowledge identities')
        for record in self.records:
            source = sources.get(record.source_id)
            if source is None or (record.hospital_name and record.hospital_name != source.publisher):
                raise ValueError('Record/source mismatch')
            if record.drug_name and source.publisher != '国家药品监督管理局':
                raise ValueError('Drug label requires a regulator source')
        return self


class KnowledgeWorkflow:
    def __init__(self, store):
        self.store = store

    def submit(self, bundle, submitter):
        timestamp = datetime.now(timezone.utc).isoformat()
        item = {'id': 'KB-'+uuid.uuid4().hex[:12], 'status': 'pending', 'submitter': submitter,
                'created_at': timestamp, 'bundle': bundle.model_dump(mode='json'), 'events': []}
        with self.store.connect() as connection:
            connection.execute('INSERT INTO knowledge VALUES (?,?,?)',
                (item['id'], 'pending', json.dumps(item, ensure_ascii=False)))
        return item

    def list(self, status=None):
        with self.store.connect() as connection:
            return [json.loads(row['payload']) for row in connection.execute(
                'SELECT payload FROM knowledge'+(' WHERE status=?' if status else '')+' ORDER BY rowid DESC',
                (status,) if status else ())]

    def review(self, identity, reviewer, approve, note):
        with self.store._lock, self.store.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT payload FROM knowledge WHERE id=?', (identity,)).fetchone()
            if row is None:
                raise LookupError('Knowledge submission not found')
            item = json.loads(row['payload'])
            if item['status'] != 'pending':
                raise ValueError('Only pending submissions can be reviewed')
            item['status'] = 'reviewed' if approve else 'rejected'
            item['events'].append({'reviewer': reviewer, 'approved': approve, 'note': note,
                                   'at': datetime.now(timezone.utc).isoformat(),
                                   'review_scope': 'source and applicability; not clinical validation'})
            connection.execute('UPDATE knowledge SET status=?,payload=? WHERE id=?',
                (item['status'], json.dumps(item, ensure_ascii=False), identity))
            return item

    def publish(self, identity, corpus, prepare_index=None):
        with self.store._lock, self.store.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT payload FROM knowledge WHERE id=?', (identity,)).fetchone()
            if row is None:
                raise LookupError('Knowledge submission not found')
            item = json.loads(row['payload'])
            if item['status'] != 'reviewed':
                raise ValueError('Source review required before publication')
            bundle = KnowledgeBundle.model_validate(item['bundle'])
            # Use the latest committed publications even if a caller holds an
            # older runtime snapshot during concurrent requests.
            from careguide.corpus import Corpus
            corpus = self.extend(Corpus())
            if {record.id for record in bundle.records} & {record.id for record in corpus.records}:
                raise ValueError('Cannot overwrite existing medical records')
            for source in bundle.sources:
                if source.id in corpus._sources and source != corpus._sources[source.id]:
                    raise ValueError('Cannot overwrite existing medical sources')
            if prepare_index is not None:
                from careguide.corpus import Corpus
                prepare_index(self.extend(Corpus(), additional=item))
            item['status'] = 'published'
            item['events'].append({'published_at': datetime.now(timezone.utc).isoformat()})
            connection.execute('UPDATE knowledge SET status=?,payload=? WHERE id=?',
                ('published', json.dumps(item, ensure_ascii=False), identity))
            return item

    def extend(self, corpus, additional=None):
        items = self.list('published')
        if additional:
            items.append(additional)
        bundles = [KnowledgeBundle.model_validate(item['bundle']) for item in sorted(items, key=lambda item: item['id'])]
        for bundle in bundles:
            corpus._sources.update({source.id: source for source in bundle.sources})
            corpus._records += tuple(bundle.records)
        if bundles:
            corpus.fingerprint = hashlib.sha256((corpus.fingerprint+json.dumps(
                [bundle.model_dump(mode='json') for bundle in bundles], sort_keys=True)).encode()).hexdigest()
        return corpus
