#!/usr/bin/env python3
"""seller_audit.py — conservative offline reanalysis of AntSeed seller probe rows.

Re-derives, from previously collected evidence only, per catalog seller:
  * tokenizer identity result: unknown / compatible_tokenizer / inconsistent_tokenizer,
  * accounting observations about the token meter (never fraud verdicts),
  * per-request availability and operational readiness,
  * one recommendation: 'avoid' (operational criterion only) or 'unknown'.

It never labels a seller HONEST / FABRICATED / swapped, never treats tokenizer
equality as model identity or trust, never infers intent or billed amounts from
usage, and never substitutes sibling references for the exact claimed model.

Usage:
  python3 seller_audit.py <rows.jsonl> --catalog <catalog.json> --refs <refs.json> \
      --out <report.json> [--markdown <report.md>]

Row dialects: legacy rows (no request_evidence) are legacy_unattributed and are
never identity evidence, even when echo_ok is true (a one-shot pin echo is not
per-request attribution). New fpx rows carry request_evidence with per-cell
{status, attribution{requested_peer, served_peer, verified}} + top-level pin_peer,
full request records in stack/reason/behav, and operational_checks
[{task, expected, status, attribution, passed, max_tokens?}].

Exit codes: 0 ok, 1 malformed input, 2 usage.
"""
import argparse, hashlib, json, statistics, sys
from datetime import datetime, timezone

PROBES = ("digits", "zh", "code", "rare", "ru", "ws", "ar", "url")
TCELLS = ("ok",) + PROBES                      # baseline "ok" cell is required too
REF_BY_CANON = {                               # exact enrolled reference per claim; NO sibling fallback
    "claude-opus-5.5": "anthropic/claude-opus-5.5", "claude-opus-5-5": "anthropic/claude-opus-5.5",
    "anthropic/claude-opus-5.5": "anthropic/claude-opus-5.5",
    "muse-wonder": "meta/muse-wonder", "meta/muse-wonder": "meta/muse-wonder",
    "kimi-k3.1-pro": "moonshotai/kimi-k3.1-pro", "kimi-k3-1-pro": "moonshotai/kimi-k3.1-pro",
    "moonshotai/kimi-k3.1-pro": "moonshotai/kimi-k3.1-pro",
    "gpt-6-astra": "openai/gpt-6-astra", "openai/gpt-6-astra": "openai/gpt-6-astra",
    "glm-5.3": "z-ai/glm-5.3", "z-ai/glm-5.3": "z-ai/glm-5.3",
    "gpt-6.1-terra": "openai/gpt-6.1-terra", "gpt-6-1-terra": "openai/gpt-6.1-terra",
    "openai/gpt-6.1-terra": "openai/gpt-6.1-terra",
}
CANON = {alias: ref.split("/", 1)[1] for alias, ref in REF_BY_CANON.items()}
POLICY_AUTH = {401, 402, 403}
RATE_LIMITED = {408, 429}

LIMITS = [
    "No seller is labelled HONEST, FABRICATED or swapped; such legacy classifications are "
    "superseded and only echoed as historical labels when present in the input.",
    "compatible_tokenizer is NOT identity proof: sibling models and gateways share tokenizers, "
    "so it never yields trust or exact-model confirmation.",
    "This tool has no independently validated exact-model calibration mode: exact-model claims "
    "stay unknown. An operational avoid recommendation does not change model identity; positive "
    "trust requires exact-model references evaluated on an independent hold-out.",
    "Rows without per-request attribution are legacy_unattributed: observations are kept, but "
    "they are not identity evidence (a one-shot echo_ok is not enough).",
    "Tokenizer comparison requires every probe cell (including the 'ok' baseline) to be status "
    "200 with verified pin attribution (requested peer == served peer) recorded per request.",
    "Accounting observations (negative deltas, negative raw counts, low or fixed counts) are "
    "metering follow-up items; no fraud, intent or billed amount is inferred.",
    "Operational avoid requires repeated fully attributed basic requests with usable token "
    "budget that produced no usable assistant content, or seller 5xx; policy/auth blocks and "
    "rate limits alone stay 'unknown'.",
    "Raw headers, error bodies, URLs and keys from source rows are summarised as status codes "
    "and reason categories only.",
]
ASSUMPTIONS = [
    "request_evidence.tok[cell] = {status, attribution{requested_peer, served_peer, verified}, "
    "raw_prompt_tokens?}; row.tok holds raw counts and features.tok holds baseline-subtracted deltas.",
    "Pinned requests use x-antseed-pin-peer and are verified by x-antseed-peer-id; x-served-by "
    "display names are never used as peer evidence.",
    "operational_checks[].max_tokens >= 64 marks a usable content budget (16-token probes "
    "cannot demonstrate content failure).",
    "Sellers found in rows but absent from the catalog are still analysed and reported "
    "(source rows_only).",
]
BLOCKERS = {"legacy_unattributed", "cells_missing", "cell_request_failed", "unverified_attribution",
            "missing_counts", "comparison_unreliable", "exact_reference_not_enrolled",
            "reference_probes_incomplete", "quarantined_accounting_only"}


def canon(model):
    return CANON.get(str(model).strip(), str(model).strip())


def parse_status(v):
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        return None
    try:
        return int(v)
    except ValueError:
        return None


def status_cat(st):
    st = parse_status(st)
    if st is None or st < 100: return "transport"
    if 200 <= st < 300: return "ok"
    if st in POLICY_AUTH: return "policy_auth"
    if st in RATE_LIMITED: return "rate_limited"
    if 500 <= st < 600: return "seller_5xx"
    return "client_4xx"


def attr_state(rec, seller_peer, pin, require_pin):
    """Verified only when recorded requested_peer == served_peer, the collector set
    verified, and the served peer is bound to this seller: to the row peerId always,
    and additionally to pin_peer for pinned requests. Never reads x-served-by."""
    if rec.get("attribution_error"):
        return "unverified", "request record marked unverified (attribution_error)"
    a = rec.get("attribution")
    if not isinstance(a, dict) or a is None: return "missing", "no attribution object"
    req, srv = a.get("requested_peer"), a.get("served_peer")
    if not req or not srv: return "missing", "attribution lacks requested/served peer"
    if srv != req: return "mismatch", "served peer differs from requested pinned peer"
    if srv != seller_peer or (pin and req != pin):
        return "mismatch", "attribution does not bind to this seller's peer"
    if a.get("verified") is not True: return "unverified", "collector did not verify served peer"
    if pin and srv != pin: return "mismatch", "served peer differs from row pin_peer"
    if require_pin and not pin: return "unverified", "row lacks pin_peer binding"
    return "verified", ""


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip(): continue
            try: r = json.loads(line)
            except json.JSONDecodeError as e: raise ValueError(f"{path} line {n}: invalid JSON ({e.msg})")
            if not isinstance(r, dict): raise ValueError(f"{path} line {n}: row is not an object")
            for k in ("model", "peerId"):
                if not isinstance(r.get(k), str) or not r[k].strip():
                    raise ValueError(f"{path} line {n}: missing or empty '{k}'")
            r["_idx"] = len(rows)
            rows.append(r)
    return rows


def load_catalog(path):
    c = json.load(open(path, encoding="utf-8"))
    if not isinstance(c, dict): raise ValueError(f"{path}: catalog must be an object")
    for model, offer in c.items():
        if not isinstance(offer, dict): raise ValueError(f"{path}: model '{model}' offer must be an object")
        peers = offer.get("peers")
        if not isinstance(peers, list): raise ValueError(f"{path}: model '{model}' needs a 'peers' list")
        for i, p in enumerate(peers):
            if not isinstance(p, dict) or not isinstance(p.get("peerId"), str) or not p["peerId"].strip():
                raise ValueError(f"{path}: model '{model}' peer #{i} missing peerId")
    return c


def load_refs(path):
    r = json.load(open(path, encoding="utf-8"))
    models = r.get("models") if isinstance(r, dict) and r.get("schema") == "fpx.reference-db/v1" else r
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected fpx.reference-db/v1 or a model map")
    vecs = {}
    for mid, m in models.items():
        if not isinstance(m, dict):
            raise ValueError(f"{path}: model '{mid}' must be an object")
        tv = []
        probes = m.get("probes")
        if probes is not None and not isinstance(probes, list):
            raise ValueError(f"{path}: model '{mid}' probes must be a list")
        for p in probes or []:
            tok = p.get("tok") if isinstance(p, dict) else None
            if isinstance(tok, dict) and all(isinstance(tok.get(k), int) and not isinstance(tok.get(k), bool) for k in PROBES):
                tv.append({k: tok[k] for k in PROBES})
        vecs[mid] = tv
    return vecs


def row_tokenizer(row, exact_vecs):
    """Per-row tokenizer evidence: state ('comparable' or a blocker), observations
    split into identity / accounting, and (when comparable) outcome + raws.
    Comparison uses the exact enrolled reference only — never siblings.
    Producer shape: request_evidence.tok[cell] = {status, attribution, error?};
    raw usage counts at row['tok'][cell]; deltas at features.tok."""
    i = row["_idx"]; feats = row.get("features") or {}
    ident, acct = [], []
    if not isinstance(row.get("request_evidence"), dict):
        for k, v in (feats.get("tok") or {}).items():
            if isinstance(v, int) and not isinstance(v, bool) and v < 0:
                acct.append({"kind": "negative_delta", "source": "legacy_reported", "cell": k, "value": v, "row_index": i})
        return {"state": "legacy_unattributed", "ident": ident, "acct": acct}
    ev = row["request_evidence"]; cells = ev.get("tok") or {}
    missing = [k for k in TCELLS if not isinstance(cells.get(k), dict)]
    if missing:
        ident.append({"kind": "cells_missing", "cells": missing, "row_index": i})
        return {"state": "cells_missing", "ident": ident, "acct": acct}
    for k in TCELLS:
        c = cells[k]; st = parse_status(c.get("status"))
        if status_cat(st) != "ok":
            ident.append({"kind": "cell_request_failed", "cell": k, "status_class": status_cat(st), "row_index": i})
            return {"state": "cell_request_failed", "ident": ident, "acct": acct}
        a, why = attr_state(c, row["peerId"], row.get("pin_peer"), require_pin=True)
        if a != "verified":
            # Quarantined/unverified cell: keep raw per-cell accounting for follow-up, never compare.
            for k2 in TCELLS:
                v = cells[k2].get("raw_prompt_tokens")
                if isinstance(v, int) and not isinstance(v, bool) and v < 0:
                    acct.append({"kind": "negative_raw", "source": "quarantined", "cell": k2, "value": v, "row_index": i})
            ident.append({"kind": f"attribution_{a}", "why": why, "cell": k, "row_index": i})
            return {"state": "unverified_attribution", "ident": ident, "acct": acct}
        if c.get("error"):
            ident.append({"kind": "cell_request_failed", "cell": k, "status_class": "ok", "row_index": i})
            return {"state": "cell_request_failed", "ident": ident, "acct": acct}
    # Quarantined cells carry raw_prompt_tokens (accounting only); identity needs row["tok"].
    raws = row.get("tok") if isinstance(row.get("tok"), dict) else {}
    base_raw = raws.get("ok")
    if not all(isinstance(raws.get(k), int) and not isinstance(raws.get(k), bool) for k in TCELLS):
        # No raw counts anywhere: nothing may reach comparison; deltas alone are not counts.
        if any(isinstance(v, int) and not isinstance(v, bool) for v in (feats.get("tok") or {}).values()):
            ident.append({"kind": "deltas_without_counts", "row_index": i})
        ident.append({"kind": "missing_counts", "row_index": i})
        return {"state": "missing_counts", "ident": ident, "acct": acct}
    vector = {k: raws[k] - base_raw for k in PROBES}
    for k in TCELLS:
        if raws[k] < 0: acct.append({"kind": "negative_raw", "source": "row_tok", "cell": k, "value": raws[k], "row_index": i})
    if 0 <= min(raws[k] for k in PROBES) < 8: acct.append({"kind": "low_count", "row_index": i, "min": min(raws[k] for k in PROBES)})
    if len({raws[k] for k in TCELLS}) == 1: acct.append({"kind": "identical_counts_across_probes", "row_index": i})
    for k, v in vector.items():
        if v < 0:
            acct.append({"kind": "negative_delta", "source": "computed", "cell": k, "value": v, "row_index": i})
    for k, v in (feats.get("tok") or {}).items():
        if isinstance(v, int) and not isinstance(v, bool) and v < 0:
            acct.append({"kind": "negative_delta", "source": "reported", "cell": k, "value": v, "row_index": i})
    outcome = None
    if exact_vecs:
        if any(all(vector[k] == r[k] for k in PROBES) for r in exact_vecs): outcome = "compatible"
        else:
            outcome = "inconsistent"
            med = min(statistics.median(abs(vector[k] - r[k]) for k in PROBES) for r in exact_vecs)
            ident.append({"kind": "closest_reference_median_abs_delta", "value": med, "row_index": i})
    return {"state": "comparable", "ident": ident, "acct": acct, "outcome": outcome, "raws": raws}


def seller_identity(rows, ref_id, vecs):
    ident, acct, outcomes, raws_all, legacy = [], [], [], [], 0
    blockers = set()
    if ref_id is None: blockers.add("exact_reference_not_enrolled")
    elif ref_id not in vecs: blockers.add("exact_reference_not_enrolled")  # sibling may be enrolled; exact is not
    elif not vecs[ref_id]: blockers.add("reference_probes_incomplete")
    exact_vecs = vecs.get(ref_id) or []
    for row in rows:
        if "operational_checks" in row and not any(k in row for k in ("tok", "features", "request_evidence")):
            continue
        t = row_tokenizer(row, exact_vecs)
        ident += t["ident"]; acct += t["acct"]
        if t["state"] == "legacy_unattributed": legacy += 1
        if t["state"] in BLOCKERS: blockers.add(t["state"])
        else: outcomes.append((row["_idx"], t["outcome"])); raws_all.append(t["raws"])
    for k in PROBES:
        vals = {r[k] for r in raws_all}
        if len(raws_all) > 1 and len(vals) > 1:
            acct.append({"kind": "variable_overhead", "cell": k, "values": sorted(vals)})
        if len({r[k] - r["ok"] for r in raws_all}) > 1:
            blockers.add("comparison_unreliable")
    if legacy and outcomes: ident.append({"kind": "mixed_eras", "legacy_rows": legacy})
    if len({o for _, o in outcomes}) > 1:
        result = "unknown"
        blockers.add("contradictory_repeats")
        ident.append({"kind": "contradictory_repeats", "outcomes": [{"row_index": i, "outcome": o} for i, o in outcomes]})
    elif outcomes and "compatible" in {o for _, o in outcomes}: result = "compatible_tokenizer"
    elif outcomes: result = "inconsistent_tokenizer"
    else: result = "unknown"
    if blockers: result = "unknown"
    state = sorted(blockers)[0] if blockers else ("comparable" if outcomes else "no_tokenizer_evidence")
    return {"result": result, "evidence_state": state,
            "reasons": sorted(blockers) or (["tokenizer_only_not_identity"] if outcomes else ["no comparable tokenizer evidence"]),
            "compared_row_indexes": [i for i, outcome in outcomes if outcome is not None],
            "legacy_unattributed_rows": legacy, "observations": ident,
            "accounting": {"observations": acct, "categories": {k: sum(1 for o in acct if o["kind"] == k) for k in {o["kind"] for o in acct}},
                           "note": "metering observations only; no conclusion about seller conduct or billed amounts"}}


def seller_operational(rows):
    checks = []
    for row in rows:
        for c in row.get("operational_checks") or []:
            if not isinstance(c, dict): raise ValueError(f"row {row['_idx']}: operational check is not an object")
            st = parse_status(c.get("status")); task = c.get("task") or "unspecified"
            a, _ = attr_state(c, row["peerId"], row.get("pin_peer"), require_pin=False)
            if a != "verified":
                checks.append({"task": task, "status": st, "cat": "unattributed"})
            elif status_cat(st) != "ok":
                checks.append({"task": task, "status": st, "cat": status_cat(st)})
            else:
                # passed=None (unfinished response) is inconclusive; only explicit False with a
                # KNOWN usable budget (>=64) can demonstrate content failure.
                budget = c.get("max_tokens")
                usable = isinstance(budget, int) and not isinstance(budget, bool) and budget >= 64
                checks.append({"task": task, "status": st,
                               "cat": ("passed" if c.get("passed") is True else
                                       "failed_content" if usable and c.get("passed") is False else
                                       "probe_inconclusive")})
    n = {k: sum(1 for c in checks if c["cat"] == k) for k in {c["cat"] for c in checks}}
    passed, failed = n.get("passed", 0), n.get("failed_content", 0) + n.get("seller_5xx", 0)
    by_task = {}
    for c in checks:
        if c["cat"] in ("failed_content", "seller_5xx"): by_task[c["task"]] = by_task.get(c["task"], 0) + 1
    # Repeats must not be aggregated into a clean verdict while failures exist: mixed repeats stay unknown.
    if passed >= 2 and passed == len(checks):
        rec, why = "usable_in_test", f"{passed} attributed checks returned correct answers with no failures recorded"
    elif by_task and max(by_task.values()) >= 2 and failed == len(checks):
        rec = "avoid_for_now"
        why = (f"repeated attributed '{max(by_task, key=by_task.get)}' checks with usable token budget "
               f"produced no usable assistant content, or seller 5xx")
    else:
        rec = "unknown"
        parts = []
        if n.get("unattributed"): parts.append(f"{n['unattributed']} check(s) lack verified attribution")
        if n.get("policy_auth") or n.get("rate_limited"): parts.append("policy/auth or rate-limit responses are not failure evidence")
        if n.get("probe_inconclusive"): parts.append("unknown or small token budgets cannot demonstrate content failure")
        if passed and failed: parts.append("repeats disagree (some pass, some fail)")
        elif n.get("seller_5xx") or n.get("failed_content"): parts.append("not enough repeated failures to justify avoid")
        why = "insufficient evidence: " + ("; ".join(parts) if parts else "no operational checks recorded")
    return {"checks_total": len(checks), "check_categories": n, "recommendation": rec, "reason": why}


def seller_availability(rows):
    cats = {}

    def bump(st): cats[status_cat(st)] = cats.get(status_cat(st), 0) + 1
    for row in rows:
        ev = row.get("request_evidence")
        if isinstance(ev, dict):
            d = (ev.get("tok") or {}).values()
            d = [v for v in d if isinstance(v, dict)]
            s = [v for v in (row.get("stack") or ev.get("stack") or {}).values() if isinstance(v, dict)]
            reason = row.get("reason") or ev.get("reason")
            r = [reason] if isinstance(reason, dict) else []
            b = [v for v in (row.get("behav") or ev.get("behav") or {}).values() if isinstance(v, dict)]
            for rec in d + s + r + b: bump(rec.get("status"))
        else:
            if row.get("skipped_probe"):
                cats["skipped"] = cats.get("skipped", 0) + 1
            es = row.get("echo_status")
            if isinstance(es, int) and not isinstance(es, bool): bump(es)
            elif row.get("error"): cats["transport"] = cats.get("transport", 0) + 1
        for c in row.get("operational_checks") or []:
            if isinstance(c, dict): bump(c.get("status"))
    return {"attempts": sum(cats.values()), "status_categories": cats}


def seller_stack_evidence(rows):
    """Independent per-record attribution validation for stack/reason/behav records."""
    n = {}
    for row in rows:
        ev = row.get("request_evidence")
        if not isinstance(ev, dict): continue
        recs = [v for v in (row.get("stack") or ev.get("stack") or {}).values() if isinstance(v, dict)]
        reason = row.get("reason") or ev.get("reason")
        if isinstance(reason, dict): recs.append(reason)
        recs += [v for v in (row.get("behav") or ev.get("behav") or {}).values() if isinstance(v, dict)]
        for rec in recs:
            a, _ = attr_state(rec, row["peerId"], row.get("pin_peer"), require_pin=False)
            n[a] = n.get(a, 0) + 1
    return {"records": sum(n.values()), "attribution_states": n, "note": "these records are never identity evidence alone"}


def recommend(op, ident, avail):
    if op["recommendation"] == "avoid_for_now":
        return {"label": "avoid", "scope": "operational",
                "reason": op["reason"] + "; exact-model identity remains unestablished"}
    parts = ["this tokenizer-only assessment cannot authenticate an exact model; "
             "behavioral reference calibration is not evaluated by this command"]
    if ident["result"] == "compatible_tokenizer":
        parts.append("tokenizer matches the claimed model's enrolled reference — compatibility only, not identity")
    elif ident["result"] == "inconsistent_tokenizer":
        parts.append("tokenizer does not match the claimed model's enrolled reference (sibling, gateway variant or "
                     "uncalibrated reference possible); re-collect with full attribution before concluding anything")
    if not avail["status_categories"].get("ok"):
        parts.append("no successful response was obtained during collection")
    if op["recommendation"] == "usable_in_test":
        parts.append("operationally usable for follow-up probing")
    return {"label": "unknown", "scope": "identity+operational", "reason": "; ".join(parts)}


def build_seller(cm, rows, peer, models, source):
    ref_id = REF_BY_CANON.get(cm)
    ident = seller_identity(rows, ref_id, models)
    op = seller_operational(rows)
    avail = seller_availability(rows)
    old = sorted({r[k] for r in rows for k in ("verdict", "legacy_verdict", "old_label") if isinstance(r.get(k), str)})
    s = {"model": cm, "exact_ref_id": ref_id, "peerId": peer.get("peerId"), "displayName": peer.get("displayName"),
         "provider": peer.get("provider"), "serviceId": peer.get("serviceId"), "source": source,
         "row_indexes": [r["_idx"] for r in rows],
         "requested_model_ids": sorted({r.get("request_model") or r["model"] for r in rows}),
         "evidence_counts": {"rows": len(rows),
                             "legacy_unattributed_rows": ident["legacy_unattributed_rows"],
                             "tokenizer_rows_compared": len(ident["compared_row_indexes"]),
                             "operational_checks": op["checks_total"], "availability_attempts": avail["attempts"]},
         "identity": ident, "availability": avail, "serving_stack_evidence": seller_stack_evidence(rows),
         "operational": op, "recommendation": recommend(op, ident, avail)}
    if old: s["superseded"] = {"legacy_labels": old, "note": "historical labels only; not used in this report"}
    return s


def audit(rows, catalog, models, sources=None):
    # Catalog offers are (model, peerId, provider, serviceId); two offers may share a peer.
    offers = []
    for model, offer in (catalog or {}).items():
        for peer in offer.get("peers") or []:
            offers.append({"model": canon(model), "raw_model": model, "peerId": peer["peerId"],
                           "displayName": peer.get("displayName"), "provider": peer.get("provider"),
                           "serviceId": peer.get("serviceId")})
    by_full, by_peer = {}, {}
    for off in offers:
        by_full.setdefault((off["model"], off["peerId"], off["provider"], off["serviceId"]), []).append(off)
        by_peer.setdefault((off["model"], off["peerId"]), []).append(off)
    groups = {}   # offer tuple -> rows bound to that exact offer
    unbound = []  # rows that could not be uniquely bound to one catalog offer
    for index, r in enumerate(rows):
        if not isinstance(r, dict): raise ValueError("row is not an object")
        for k in ("model", "peerId"):
            if not isinstance(r.get(k), str) or not r[k].strip(): raise ValueError(f"row: missing or empty '{k}'")
        r["_idx"] = index
        cm = canon(r["model"])
        rp, rprov, rsvc = r["peerId"], r.get("provider"), r.get("serviceId")
        if isinstance(rprov, str) and isinstance(rsvc, str) and (cm, rp, rprov, rsvc) in by_full:
            key = (cm, rp, rprov, rsvc)                     # exact four-key match
        elif not isinstance(rprov, str) and not isinstance(rsvc, str) and len(by_peer.get((cm, rp), [])) == 1:
            key = (cm, rp, by_peer[(cm, rp)][0]["provider"], by_peer[(cm, rp)][0]["serviceId"])
        else:
            unbound.append((r, (cm, rp, rprov if isinstance(rprov, str) else None,
                                rsvc if isinstance(rsvc, str) else None)))
            continue
        groups.setdefault(key, []).append(r)
    unbound_by_key = {}
    for r, key in unbound:
        # Legacy ambiguity: an unbound row must not contaminate any offer; keep under its
        # own (model, peer) bucket, matched at report time to whichever offer it names.
        unbound_by_key.setdefault(key, []).append(r)
    sellers = []
    for key, offs in sorted(by_full.items()):
        off = offs[0]
        grp = groups.get(key, [])
        sellers.append(build_seller(key[0], grp, {"peerId": key[1], "displayName": off["displayName"],
                                                  "provider": key[2], "serviceId": key[3]}, models, "catalog"))
    for (cm, rp, rprov, rsvc), grp in sorted(unbound_by_key.items(),
            key=lambda item: tuple(v or "" for v in item[0])):
        r0 = grp[0]
        sellers.append(build_seller(cm, grp, {"peerId": rp, "displayName": r0.get("displayName"),
                                              "provider": rprov, "serviceId": rsvc}, models, "rows_only"))
    counts = {"sellers": len(sellers),
              "recommendation": {k: sum(1 for s in sellers if s["recommendation"]["label"] == k) for k in ("trust", "avoid", "unknown")},
              "identity_result": {k: sum(1 for s in sellers if s["identity"]["result"] == k)
                                  for k in ("unknown", "compatible_tokenizer", "inconsistent_tokenizer")},
              "operational": {k: sum(1 for s in sellers if s["operational"]["recommendation"] == k)
                              for k in ("usable_in_test", "avoid_for_now", "unknown")},
              "identity_untestable_sellers": sum(1 for s in sellers if s["identity"]["evidence_state"] in BLOCKERS),
              "unavailable_sellers": sum(1 for s in sellers if not s["availability"]["status_categories"].get("ok")),
              "rows_only_sellers": sum(1 for s in sellers if s["source"] == "rows_only")}
    report = {"tool": "seller_audit", "version": 1, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "sources": sources or {}, "method": "offline reanalysis of stored rows; makes no network calls",
              "limits": LIMITS, "assumptions": ASSUMPTIONS, "counts": counts, "sellers": sellers}
    return report, markdown(report)


def md_esc(v):
    s = "" if v is None else str(v)
    return (s.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`").replace("*", "\\*")
            .replace("\r", " ").replace("\n", " "))


def markdown(rep):
    c = rep["counts"]; L = ["# Seller trust reanalysis (offline)", "",
                            f"Generated: {md_esc(rep['generated_at'])}",
                            "Sources: " + "; ".join(f"{k} `{md_esc(v)}" + ("`" if v else "") for k, v in rep["sources"].items()), "",
                            "## Limits", *(f"- {x}" for x in rep["limits"]), "", "## Assumptions",
                            *(f"- {x}" for x in rep["assumptions"]), "", "## Totals",
                            f"- sellers: {c['sellers']}; recommendations: trust {c['recommendation']['trust']}, "
                            f"avoid {c['recommendation']['avoid']}, unknown {c['recommendation']['unknown']}",
                            f"- identity: unknown {c['identity_result']['unknown']}, compatible_tokenizer "
                            f"{c['identity_result']['compatible_tokenizer']}, inconsistent_tokenizer {c['identity_result']['inconsistent_tokenizer']}",
                            f"- identity-untestable sellers: {c['identity_untestable_sellers']}; unavailable sellers: "
                            f"{c['unavailable_sellers']}; rows-only sellers: {c['rows_only_sellers']}", "", "## Follow-up"]
    acts = []
    for s in rep["sellers"]:
        who = f"{md_esc(s['model'])} / {md_esc(s['displayName'])} / `{md_esc(s['peerId'])}`"
        if s["recommendation"]["label"] == "avoid": acts.append(f"- **avoid**: {who} — {s['recommendation']['reason']}")
        if s["identity"]["accounting"]["categories"]: acts.append(f"- **metering**: {who} — {sorted(s['identity']['accounting']['categories'])}")
        if s["availability"]["attempts"] == 0: acts.append(f"- **recollect**: {who} — no recorded attempts")
        elif not s["availability"]["status_categories"].get("ok"): acts.append(f"- **recollect**: {who} — no successful attempt")
        if s["identity"]["evidence_state"] in ("unverified_attribution", "cells_missing", "cell_request_failed"):
            acts.append(f"- **recollect**: {who} — {s['identity']['evidence_state']}")
        if s["identity"]["evidence_state"] == "legacy_unattributed": acts.append(f"- **recollect**: {who} — legacy rows lack per-request attribution")
        if s["operational"]["checks_total"] == 0: acts.append(f"- **probe**: {who} — no operational checks recorded")
        elif s["identity"]["result"] == "compatible_tokenizer":
            acts.append(f"- **calibrate**: {who} — tokenizer-compatible; include in a future independently calibrated exact-model hold-out")
    L += acts or ["- none"]
    L += ["", "## Per-seller", "", "| Model | Seller | Peer | Identity | Evidence | Accounting | Operational | Recommendation |",
          "|---|---|---|---|---|---|---|---|"]
    for s in rep["sellers"]:
        e = s["evidence_counts"]
        L.append("| " + " | ".join(md_esc(x) for x in [
            s["model"], s["displayName"], s["peerId"], s["identity"]["result"],
            f"rows {e['rows']}, legacy {e['legacy_unattributed_rows']}, tok-compared {e['tokenizer_rows_compared']}, "
            f"ops-checks {e['operational_checks']}, attempts {e['availability_attempts']}",
            ", ".join(sorted(s["identity"]["accounting"]["categories"])) or "-", s["operational"]["recommendation"],
            s["recommendation"]["label"]]) + " |")
    return "\n".join(L) + "\n"


def load(path):
    with open(path, encoding="utf-8") as f: return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="seller_audit.py", description="Conservative offline seller trust reanalysis.")
    ap.add_argument("rows", help="rows.jsonl (legacy and/or request_evidence rows)")
    ap.add_argument("--catalog", required=True); ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--markdown")
    a = ap.parse_args(argv)
    try:
        rows, catalog, models = load_rows(a.rows), load_catalog(a.catalog), load_refs(a.refs)
        sources = {"rows": a.rows, "catalog": a.catalog, "refs": a.refs}
        for k, p in (("rows_sha256", a.rows), ("catalog_sha256", a.catalog), ("refs_sha256", a.refs)):
            try: sources[k] = hashlib.sha256(load_bytes(p)).hexdigest()[:16]
            except OSError: pass
        report, md = audit(rows, catalog, models, sources)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"seller_audit: {e}", file=sys.stderr); return 1
    with open(a.out, "w", encoding="utf-8") as f: json.dump(report, f, indent=1)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as f: f.write(md)
    c = report["counts"]
    print(f"seller_audit: {c['sellers']} sellers -> trust {c['recommendation']['trust']}, avoid "
          f"{c['recommendation']['avoid']}, unknown {c['recommendation']['unknown']}")
    return 0


def load_bytes(p):
    with open(p, "rb") as f: return f.read()


if __name__ == "__main__":
    sys.exit(main())