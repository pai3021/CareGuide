# CareGuide notices and third-party materials

CareGuide includes project code, evaluation assets, and redistributed research
data. General Agent engineering modules have development lineage through
EchoMind, NexusOps, and OrbisDesk. The external projects and datasets used or
referenced by CareGuide are listed below.

## Data, models, and integrations

- **[CBLUE / KUAKE](https://github.com/CBLUEbenchmark/CBLUE)** is Apache-2.0 in the pinned dataset manifests.
  CareGuide includes normalized KUAKE-QIC data and a deterministic KUAKE-IR
  derivative with source revisions, checksums, split policy, and usage scope.
- **[Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite)** is published by FreedomIntelligence under Apache-2.0.
  CareGuide includes a deterministic 1,000-row research slice with attribution,
  upstream revision, raw checksum, and sampling seed.
- **[Tencent WeKnora](https://github.com/Tencent/WeKnora)** is MIT licensed. CareGuide does not vendor its source and
  only provides an optional MCP-compatible knowledge backend.
- **[H-AdminSim](https://github.com/ljm565/H-AdminSim)** is Apache-2.0 licensed. It informs synthetic administrative
  workflow design; its runtime is not vendored.
- **[HAPI FHIR](https://github.com/hapifhir/hapi-fhir-jpaserver-starter)** runs as a separately versioned Apache-2.0 container with synthetic data.
- **[PatientAgentBench](https://github.com/amazon-science/PatientAgentBench)** is CC BY-NC 4.0. CareGuide references evaluation
  dimensions without redistributing its code, prompts, datasets, or scores.
- `hfl/chinese-roberta-wwm-ext`, Qwen embedding/reranker models, MiMo, and
  DeepSeek remain subject to their providers' model terms and service policies.

## Repository boundaries

- Model checkpoints, provider credentials, Chroma stores, traces, and generated
  evaluation outputs are excluded from Git.
- Public medical passages are research data and have not been clinically
  reviewed by CareGuide.
- Synthetic hospital resources do not represent real clinicians, availability,
  appointments, or clinical systems.
