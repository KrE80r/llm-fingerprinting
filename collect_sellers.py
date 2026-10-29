"""Collect seller-attributed readiness checks; these do not authenticate model identity."""
import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import urllib.request

import fpx

TARGETS = ("claude-opus-5.5", "muse-wonder", "kimi-k3.1-pro", "gpt-6-astra", "glm-5.3", "gpt-6.1-terra")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def select_models(document, models):
    """Match explicit catalog ids/aliases, never fast/pro/flash siblings by substring."""
    if not isinstance(document, dict) or not isinstance(document.get("data"), list):
        raise ValueError("catalog must contain a data array")
    result = {}
    for target in models:
        matches = []
        for model in document["data"]:
            if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                raise ValueError("invalid catalog model")
            aliases = model.get("aliases") or []
            if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
                raise ValueError("invalid catalog aliases")
            if target in [model["id"], *aliases]:
                matches.append(model)
        if len(matches) > 1:
            raise ValueError(f"ambiguous catalog alias: {target}")
        if not matches:
            result[target] = {"name": target, "catalog_id": None, "peers": [], "missing": True}
            continue
        model = matches[0]
        peers = model.get("peers")
        if not isinstance(peers, list):
            raise ValueError(f"missing peers array: {target}")
        seen = set()
        offers = []
        for peer in peers:
            if not isinstance(peer, dict) or any(not isinstance(peer.get(k), str) or not peer[k]
                                                 for k in ("peerId", "provider", "serviceId")):
                raise ValueError(f"invalid peer offer: {target}")
            key = (peer["peerId"], peer["provider"], peer["serviceId"])
            if key not in seen:
                seen.add(key)
                offers.append(peer)
        result[target] = {"name": model.get("name", target), "catalog_id": model["id"], "peers": offers}
    return result


def check_result(record, expected):
    """A finite-budget unfinished response is unknown, not an incorrect model answer."""
    if not (record.get("attribution") or {}).get("verified") or record.get("status") != 200:
        return None
    document = record.get("json")
    if not isinstance(document, dict):
        return None
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    if choice.get("finish_reason") not in ("stop", "end_turn"):
        return None
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        return None
    return message["content"].strip() == expected


def run_offer(base, key, model, peer, timeout):
    row = {"schema": "fpx.seller-observation/v1", "model": model,
           "request_model": peer["serviceId"], "peerId": peer["peerId"],
           "displayName": peer.get("displayName", peer["peerId"]), "provider": peer["provider"],
           "serviceId": peer["serviceId"], "started_at": utcnow(), "operational_checks": []}
    nonce = secrets.token_hex(6)
    a, b = secrets.randbelow(9000) + 1000, secrets.randbelow(9000) + 1000
    tasks = [("echo", f"Reply with exactly this text, nothing else: {nonce}", nonce),
             ("arithmetic", f"What is {a} + {b}? Reply with digits only. Check tag: {nonce}", str(a + b))]
    for task, prompt, expected in tasks:
        record = fpx._chat(base.rstrip("/") + "/chat/completions", key, peer["serviceId"],
                           [{"role": "user", "content": prompt}], max_tokens=1024,
                           pin=peer["peerId"], timeout=timeout)
        record.update(task=task, prompt=prompt, expected=expected, max_tokens=1024, checked_at=utcnow(),
                      passed=check_result(record, expected))
        # Keep only routing/request identifiers, never Set-Cookie or response credentials.
        record["headers"] = {k: v for k, v in (record.get("headers") or {}).items()
                             if k.lower() in ("x-antseed-peer-id", "x-antseed-request-id", "x-request-id")}
        row["operational_checks"].append(record)
    row["finished_at"] = utcnow()
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="OpenAI-compatible router API base")
    parser.add_argument("--key-env")
    parser.add_argument("--models", nargs="+", default=TARGETS)
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory for private raw evidence")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--timeout", type=int, choices=range(1, 181), default=45)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        parser.error("out-dir already exists; use a new directory to preserve prior observations")
    if args.key_env and not os.environ.get(args.key_env):
        parser.error("specified key environment variable is not set")
    key = os.environ.get(args.key_env, "") if args.key_env else ""
    headers = {"authorization": "Bearer " + key} if key else {}
    request = urllib.request.Request(args.base.rstrip("/") + "/models", headers=headers)
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        raw_catalog = json.load(response)
    catalog = select_models(raw_catalog, args.models)
    args.out_dir.mkdir(parents=True, mode=0o700)
    (args.out_dir / "catalog.raw.json").write_text(json.dumps(raw_catalog, indent=2), encoding="utf-8")
    (args.out_dir / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    slots = [(model, peer) for model, meta in catalog.items() for peer in meta["peers"]]
    print(f"Checking {len(slots)} advertised offers with two requests each; model identity remains unverified.", flush=True)
    with (args.out_dir / "rows.jsonl").open("w", encoding="utf-8") as output, cf.ThreadPoolExecutor(args.workers) as executor:
        futures = {executor.submit(run_offer, args.base, key, model, peer, args.timeout): (model, peer)
                   for model, peer in slots}
        for index, future in enumerate(cf.as_completed(futures), 1):
            model, peer = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {"model": model, "peerId": peer["peerId"], "displayName": peer.get("displayName", peer["peerId"]),
                       "provider": peer["provider"], "serviceId": peer["serviceId"],
                       "operational_checks": [], "collection_error": type(error).__name__}
            output.write(json.dumps(row) + "\n")
            output.flush()
            checks = row["operational_checks"]
            passed = sum(c.get("passed") is True for c in checks)
            verified = sum(bool((c.get("attribution") or {}).get("verified")) for c in checks)
            print(f"{index}/{len(slots)} {model} {peer['peerId'][:12]}: attributed={verified}/2 passed={passed}/2", flush=True)


if __name__ == "__main__":
    main()
