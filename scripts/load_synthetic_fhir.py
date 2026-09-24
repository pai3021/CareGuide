"""Validate or explicitly upload the frozen synthetic FHIR transaction bundle."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--apply", action="store_true")
    parser.add_argument("--url", default=os.environ.get("CAREGUIDE_FHIR_BASE_URL", "http://127.0.0.1:8080/fhir"))
    args = parser.parse_args()
    bundle = json.loads((ROOT/"data/synthetic_hospital/fhir_bundle.json").read_text(encoding="utf-8"))
    assert bundle["resourceType"] == "Bundle" and bundle["type"] == "transaction"
    print(json.dumps({"valid": True, "resource_count": len(bundle["entry"]), "applied": args.apply}, ensure_ascii=False))
    if args.apply:
        hostname = (urlsplit(args.url).hostname or '').casefold()
        # Windows developer shells often define an HTTP proxy without a
        # matching NO_PROXY entry. Local HAPI traffic must stay local.
        with httpx.Client(trust_env=hostname not in {'127.0.0.1', 'localhost'}) as client:
            response = client.post(args.url.rstrip("/"), json=bundle,
                                   headers={"Content-Type":"application/fhir+json"}, timeout=60)
            response.raise_for_status()


if __name__ == "__main__": main()
