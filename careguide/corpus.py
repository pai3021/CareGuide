"""Validate corpus integrity, authority, scope and source review evidence."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / 'data' / 'medical_v1'
AUTHORITIES = {
    'www.nmpa.gov.cn': '国家药品监督管理局',
    'www.pkuih.edu.cn': '北京大学国际医院',
    'www.pkufh.com': '北京大学第一医院',
    'www.nhs.uk': '英国 NHS',
    'www.njzdyy.com': '东南大学附属中大医院',
    'huggingface.co': 'FreedomIntelligence',
    'tianchi.aliyun.com': 'CBLUE / Alibaba KUAKE',
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class Source(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str
    publisher: str
    jurisdiction: Literal['CN', 'GB']
    scope: str = Field(min_length=1)
    published_at: date | None
    verified_at: date
    valid_until: date
    review_status: Literal['developer_source_verified', 'open_dataset_unreviewed']
    clinical_reviewed: Literal[False]
    verification_method: Literal['developer_html_source_verification',
                                 'developer_visual_pdf_verification',
                                 'dataset_revision_and_checksum']
    locator: str = Field(min_length=1)
    retrieved_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    evidence_anchors: tuple[str, ...] = Field(min_length=1)
    license: str | None = None
    dataset_revision: str | None = None

    @model_validator(mode='after')
    def validate_authority_and_dates(self):
        url = urlparse(self.url)
        if (url.scheme != 'https' or url.username or url.password or url.port
                or AUTHORITIES.get(url.hostname) != self.publisher):
            raise ValueError('Source must use the declared official HTTPS authority')
        if self.review_status == 'open_dataset_unreviewed':
            if not self.license or not self.dataset_revision:
                raise ValueError('Open dataset sources require license and revision')
        elif self.license is not None or self.dataset_revision is not None:
            raise ValueError('Dataset metadata is only valid for open dataset sources')
        if self.valid_until < self.verified_at:
            raise ValueError('Source expiry precedes verification')
        if self.published_at and self.published_at > self.verified_at:
            raise ValueError('Source publication is later than verification')
        return self


class DocumentKind(str, Enum):
    CARE_GUIDANCE = 'care_guidance'
    DEPARTMENT = 'department'
    DRUG_LABEL = 'drug_label'
    DRUG_INTERACTION = 'drug_interaction'
    HOSPITAL_GUIDE = 'hospital_guide'


class Record(StrictModel):
    id: str = Field(min_length=1)
    source_id: str
    kind: DocumentKind
    title: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=1000)
    keywords: tuple[str, ...] = Field(min_length=1)
    representation: Literal['curated_factual_summary', 'open_dataset_qa']
    locator: str | None = None
    hospital_id: str | None = None
    hospital_name: str | None = None
    campus: str | None = None
    department: str | None = None
    drug_name: str | None = None
    dosage_form: str | None = None
    specification: str | None = None
    label_type: Literal['regulatory_template'] | None = None
    section: str | None = None
    other_drug: str | None = None

    @model_validator(mode='after')
    def validate_scope(self):
        if self.kind in {DocumentKind.DEPARTMENT, DocumentKind.HOSPITAL_GUIDE}:
            if not all([self.hospital_id, self.hospital_name, self.campus]):
                raise ValueError('Hospital documents require hospital and campus scope')
        if self.kind == DocumentKind.DEPARTMENT and not self.department:
            raise ValueError('Department document requires department identity')
        if self.kind in {DocumentKind.DRUG_LABEL, DocumentKind.DRUG_INTERACTION}:
            if not all([self.drug_name, self.dosage_form, self.specification,
                        self.label_type, self.section, self.locator]):
                raise ValueError('Medicine documents require exact product and section scope')
        if self.kind == DocumentKind.DRUG_INTERACTION and not self.other_drug:
            raise ValueError('Interaction document requires explicit second substance')
        if self.representation == 'open_dataset_qa' and self.kind != DocumentKind.CARE_GUIDANCE:
            raise ValueError('Open dataset QA is limited to general care guidance')
        return self


class Corpus:
    def __init__(self, root: Path | None = None):
        root = Path(root or os.environ.get('CAREGUIDE_CORPUS_DIR') or DEFAULT_CORPUS)
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['version'] not in {'careguide-medical-corpus-v1', 'careguide-medical-zhongda-v1',
                                        'careguide-medical-zhongda-v2',
                                        'careguide-open-medical-v1',
                                        'careguide-kuake-rag-v1'}:
            raise ValueError('Unsupported medical corpus version')
        self.version = manifest['version']
        if set(manifest['files']) != {'sources.json', 'records.json'}:
            raise ValueError('Unexpected corpus files')
        for filename, expected in manifest['files'].items():
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Corpus integrity mismatch: {filename}')
        sources = [Source.model_validate(s) for s in json.loads((root/'sources.json').read_text(encoding='utf-8'))]
        records = [Record.model_validate(r) for r in json.loads((root/'records.json').read_text(encoding='utf-8'))]
        if len({s.id for s in sources}) != len(sources) or len({r.id for r in records}) != len(records):
            raise ValueError('Duplicate corpus identities')
        self._sources = {s.id: s for s in sources}
        self._records = tuple(records)
        if len(sources) != manifest['source_count'] or len(records) != manifest['record_count']:
            raise ValueError('Corpus counts disagree with manifest')
        for record in records:
            source = self._sources.get(record.source_id)
            if not source:
                raise ValueError('Record references unknown source')
            if record.hospital_name and record.hospital_name != source.publisher:
                raise ValueError('Hospital record disagrees with source publisher')
            if record.drug_name and source.publisher != '国家药品监督管理局':
                raise ValueError('Seed medicine notes require the verified regulator source')
        self.fingerprint = hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()

    @property
    def records(self) -> tuple[Record, ...]:
        return self._records

    def fresh(self, record: Record, today: date) -> bool:
        source = self._sources[record.source_id]
        return source.verified_at <= today <= source.valid_until

    def evidence(self, record: Record) -> dict:
        source = self._sources[record.source_id]
        return {'record_id': record.id, 'kind': record.kind.value,
                'title': record.title, 'content': record.content,
                'representation': record.representation,
                'scope': {key: value for key, value in record.model_dump(mode='json').items()
                          if key in {'hospital_id','hospital_name','campus','department','drug_name',
                                     'dosage_form','specification','label_type','section','other_drug'}
                          and value is not None},
                'citation': {'source_id': source.id, 'url': source.url,
                             'publisher': source.publisher, 'jurisdiction': source.jurisdiction,
                             'source_scope': source.scope, 'published_at': source.model_dump(mode='json')['published_at'],
                             'verified_at': source.verified_at.isoformat(),
                             'valid_until': source.valid_until.isoformat(),
                             'review_status': source.review_status,
                             'clinical_reviewed': source.clinical_reviewed,
                             'license': source.license,
                             'dataset_revision': source.dataset_revision,
                             'locator': record.locator or source.locator,
                             'retrieved_sha256': source.retrieved_sha256,
                             'content_sha256': hashlib.sha256(record.content.encode()).hexdigest()}}

    def coverage(self, today: date) -> dict:
        return {'version': self.version, 'fingerprint': self.fingerprint,
                'source_count': len(self._sources), 'record_count': len(self._records),
                'active_record_count': sum(self.fresh(r, today) for r in self._records),
                'clinical_reviewed': False,
                'hospitals': sorted({(r.hospital_id, r.hospital_name, r.campus)
                                     for r in self._records if r.hospital_id}),
                'products': sorted({(r.drug_name, r.dosage_form, r.specification)
                                    for r in self._records if r.drug_name}),
                'interaction_coverage': 'Only explicit seed label entries; no complete interaction database'}
