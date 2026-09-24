"""Download and integrity-check the official KUAKE-IR file layout.

The Tianchi download page is the authoritative source.  Its interactive
download is mirrored at a pinned Git commit for reproducible transport.  Raw
benchmark files stay in ``data/source_cache`` and are never committed; only a
small provenance manifest is written under ``data/external``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "source_cache" / "kuake_ir"
MANIFEST_DIR = ROOT / "data" / "external" / "kuake_ir"
REVISION = "e477490616e981b1a6dfcd99ddca814577eab1be"
REPOSITORY = "githuBlijingai/KUAKE-LR-Rerank"
EXPECTED = {
    "corpus.tsv": (960_363, "2921f0ba0854b40e8f607c75d232cf9dfd6afad11fc759fc46f93560961933bf"),
    "KUAKE-IR_train_query.txt": (99_999, "20478c6832bb26fb84b89da7ba9f9d511ad843f6d4148641b95e90da65d3de1b"),
    "KUAKE-IR_train.tsv": (99_999, "cc62ddc16430f960ec48a273ed719b3a41718ce1ae32abde179b864ffdbedc1b"),
    "KUAKE-IR_dev_query.txt": (1_000, "c8d9bfb6779d2e29939a6e8d7dee675a8be5a18876d9a478645279261411fd28"),
    "KUAKE-IR_dev.tsv": (1_000, "47cfded0ba119f720f5d04c2a30abd3f48fce9a51928dfd5f7207e261974fce7"),
    "KUAKE-IR_test.query.txt": (3_000, "ec43c991f78e6ed18ab7c4608561319272106bfbc4fd251e0cc034ad129a8788"),
    "KUAKE-IR_test.tsv": (3_000, "2e6f85a17ec7c1769253447ae2b84f6ef3f00a43cad1a9f5904dce638ea53d56"),
    "README.txt": (10, "20978fb011519e2c055595d33e95a0db570e4c8b8fa7eee9ed4fed5d4d5afd50"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _url(name: str) -> str:
    host = "media.githubusercontent.com/media" if name == "corpus.tsv" else "raw.githubusercontent.com"
    return f"https://{host}/{REPOSITORY}/{REVISION}/KUAKE-IR/{name}"


def download(path: Path) -> None:
    request = Request(_url(path.name), headers={"User-Agent": "CareGuide-dataset-import/1.0"})
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(request, timeout=180) as source, temporary.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    temporary.replace(path)


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def read_pairs(path: Path, *, allow_empty_value: bool = False):
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            value = line.rstrip("\r\n")
            parts = value.split("\t", 1)
            if len(parts) == 1 and allow_empty_value:
                parts.append("")
            parts[0] = parts[0].strip()
            if allow_empty_value:
                parts[1] = parts[1].strip()
            if len(parts) != 2 or not parts[0] or (not parts[1] and not allow_empty_value):
                raise ValueError(f"Malformed row in {path.name}:{number}")
            yield parts[0], parts[1]


def validate_relationships(source: Path) -> None:
    corpus_ids = {key for key, _ in read_pairs(source / "corpus.tsv")}
    if len(corpus_ids) != EXPECTED["corpus.tsv"][0]:
        raise ValueError("KUAKE-IR corpus IDs are not unique")
    for split, query_name, relation_name in (
        ("train", "KUAKE-IR_train_query.txt", "KUAKE-IR_train.tsv"),
        ("dev", "KUAKE-IR_dev_query.txt", "KUAKE-IR_dev.tsv"),
    ):
        query_ids = {key for key, _ in read_pairs(source / query_name)}
        relations = dict(read_pairs(source / relation_name))
        if query_ids != set(relations):
            raise ValueError(f"KUAKE-IR {split} query/relation IDs disagree")
        missing = set(relations.values()) - corpus_ids
        if missing:
            raise ValueError(f"KUAKE-IR {split} references missing corpus IDs")
    test_ids = {key for key, _ in read_pairs(source / "KUAKE-IR_test.query.txt")}
    test_relations = dict(read_pairs(source / "KUAKE-IR_test.tsv", allow_empty_value=True))
    if test_ids != set(test_relations) or any(value.strip() for value in test_relations.values()):
        raise ValueError("KUAKE-IR test relation placeholders are malformed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    source.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict] = {}
    for name, (expected_count, expected_hash) in EXPECTED.items():
        path = source / name
        if not path.exists() and args.download:
            download(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; rerun with --download")
        actual_hash, actual_count = sha256(path), line_count(path)
        if actual_hash != expected_hash or actual_count != expected_count:
            raise ValueError(f"Integrity mismatch for {name}")
        files[name] = {"line_count": actual_count, "sha256": actual_hash}

    validate_relationships(source)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "careguide-kuake-ir-manifest-v1",
        "dataset": "CBLUE KUAKE-IR",
        "license": "Apache-2.0",
        "license_file": "../LICENSE-APACHE-2.0.txt",
        "official_source": "https://github.com/CBLUEbenchmark/CBLUE",
        "official_download_page": "https://tianchi.aliyun.com/dataset/95414",
        "transport_mirror": {"repository": REPOSITORY, "revision": REVISION},
        "raw_storage": "data/source_cache/kuake_ir (gitignored)",
        "usage": {"train": "retrieval_development", "dev": "external_retrieval_evaluation",
                  "test": "unlabeled_upstream_test_not_scored_locally"},
        "files": files,
    }
    (MANIFEST_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (MANIFEST_DIR / "README.md").write_text(
        "# KUAKE-IR 外部检索基准\n\n"
        "本目录只提交来源与完整性 manifest。约350MB原始语料、查询和相关性文件保存在"
        "`data/source_cache/kuake_ir`，不会提交 Git。Tianchi/CBLUE 是权威来源；固定 Git 提交"
        "只作可重复传输。train 用于检索开发，dev 用于外部评测，test 没有公开标签。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
