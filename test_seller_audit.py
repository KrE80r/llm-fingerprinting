"""Tests for seller_audit.py: consumer-visible decision errors, offline stdlib only."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "seller_audit.py")
MODEL = "claude-opus-5.5"
REF_ID = "anthropic/claude-opus-5.5"
REF_VEC = {"digits": 26, "zh": 36, "code": 43, "rare": 78, "ru": 64, "ws": 11, "ar": 57, "url": 34}
PEER = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def attr(peer=PEER, verified=True, served=None):
    served = served or peer
    return {"requested_peer": peer, "served_peer": served, "verified": verified}


def cell(code, a=None, raw=None, error=None):
    c = {"status": code}
    if error: c["error"] = error
    if raw is not None: c["prompt_tokens"] = raw
    if a is not None: c["attribution"] = a
    return c


def ev_row(pin=PEER, raws=None, deltas=None):
    """New-dialect row: request_evidence with per-cell status/attribution; raw usage
    counts at row['tok'] (ok baseline + 8 probes); deltas reported at features.tok."""
    raws = raws or {k: 9 + v for k, v in REF_VEC.items()}
    raws = {"ok": raws.get("ok", 9), **{k: raws.get(k, 9 + REF_VEC[k]) for k in REF_VEC}}
    tok_cell = {k: cell(200, attr()) for k in raws}
    checks = []
    row = {"model": MODEL, "peerId": PEER, "displayName": "Test Seller",
           "pin_peer": pin, "pin_ok": True,
           "tok": raws,
           "request_evidence": {"tok": tok_cell},
           "operational_checks": checks}
    if deltas: row["features"] = {"tok": deltas}
    return row


def run(rows, catalog, refs, extra=()):
    with tempfile.TemporaryDirectory() as d:
        rp, cp, fp = (os.path.join(d, n) for n in ("rows.jsonl", "catalog.json", "refs.json"))
        Path(rp).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        Path(cp).write_text(json.dumps(catalog), encoding="utf-8")
        Path(fp).write_text(json.dumps(refs), encoding="utf-8")
        out = os.path.join(d, "report.json")
        p = subprocess.run([sys.executable, TOOL, rp, "--catalog", cp, "--refs", fp, "--out", out, *extra],
                           capture_output=True, text=True)
        rep = json.loads(Path(out).read_text(encoding="utf-8")) if p.returncode == 0 else None
        return p.returncode, p.stderr, rep


def refs_db():
    return {"schema": "fpx.reference-db/v1", "models": {REF_ID: {"probes": [{"tok": REF_VEC}]}}}


CATALOG = {MODEL: {"peers": [{"peerId": PEER, "displayName": "Test Seller", "provider": "openai", "serviceId": MODEL}]}}
SIBLING_REFS = {"schema": "fpx.reference-db/v1",
                "models": {"anthropic/claude-sonnet-5.5": {"probes": [{"tok": {k: REF_VEC[k] + 5 for k in REF_VEC}}]}}}


def seller0(rep):
    return rep["sellers"][0]


class Baseline(unittest.TestCase):
    def test_fully_attributed_compatible_is_still_unknown(self):
        """Tokenizer equality against the exact enrolled reference NEVER yields trust."""
        code, _, rep = run([ev_row()], CATALOG, refs_db())
        self.assertEqual(code, 0)
        s = seller0(rep)
        self.assertEqual(s["identity"]["result"], "compatible_tokenizer")
        self.assertEqual(s["recommendation"]["label"], "unknown")
        # compatible tokenizer must not flip the top-level recommendation away from unknown
        self.assertEqual(rep["counts"]["recommendation"]["trust"], 0)

    def test_legacy_rows_are_legacy_unattributed_not_identity(self):
        """Old rows with echo_ok=true and no request_evidence cannot back identity."""
        row = {"model": MODEL, "peerId": PEER, "echo_ok": True, "echo_status": 200, "echo_peer": PEER,
               "features": {"tok": dict(REF_VEC)}}
        code, _, rep = run([row], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(s["identity"]["result"], "unknown")
        self.assertEqual(s["identity"]["evidence_state"], "legacy_unattributed")
        self.assertEqual(s["recommendation"]["label"], "unknown")

    def test_sibling_only_refs_do_not_become_compatible(self):
        """Sibling is enrolled, exact ref is absent: unknown, no proxy fallback."""
        code, _, rep = run([ev_row()], CATALOG, SIBLING_REFS)
        s = seller0(rep)
        self.assertEqual(s["identity"]["result"], "unknown")
        self.assertEqual(s["identity"]["evidence_state"], "exact_reference_not_enrolled")
        self.assertIn("exact_reference_not_enrolled", s["identity"]["reasons"])

    def test_no_sibling_fallback_even_when_sibling_matches(self):
        """Row matches the sibling vector but the exact ref exists: inconsistent vs exact."""
        sib = {k: REF_VEC[k] + 5 for k in REF_VEC}
        refs = refs_db(); refs["models"]["anthropic/claude-sonnet-5.5"] = {"probes": [{"tok": sib}]}
        code, _, rep = run([ev_row(raws={k: 9 + v for k, v in sib.items()})], CATALOG, refs)
        s = seller0(rep)
        self.assertEqual(s["identity"]["result"], "inconsistent_tokenizer")
        self.assertEqual(s["recommendation"]["label"], "unknown")

    def test_readiness_rows_do_not_invalidate_separate_tokenizer_evidence(self):
        readiness = {"model": MODEL, "peerId": PEER, "operational_checks": []}
        code, _, rep = run([readiness, ev_row()], CATALOG, refs_db())
        self.assertEqual(code, 0)
        seller = seller0(rep)
        self.assertEqual(seller["identity"]["result"], "compatible_tokenizer")
        self.assertEqual(seller["evidence_counts"]["tokenizer_rows_compared"], 1)
        self.assertEqual(seller["evidence_counts"]["legacy_unattributed_rows"], 0)

    def test_top_level_fpx_records_are_included_in_attribution_and_availability(self):
        row = ev_row()
        row["stack"] = {"sys": {"status": 200, "attribution": attr()}}
        row["reason"] = {"status": 200, "attribution": attr(served="other-peer")}
        row["behav"] = {"echo": {"status": 403}}
        code, _, rep = run([row], CATALOG, refs_db())
        self.assertEqual(code, 0)
        seller = seller0(rep)
        self.assertEqual(seller["serving_stack_evidence"]["attribution_states"],
                         {"verified": 1, "mismatch": 1, "missing": 1})
        self.assertEqual(seller["availability"]["attempts"], 12)
        self.assertEqual(seller["availability"]["status_categories"]["policy_auth"], 1)

    def test_producer_attribution_error_retains_quarantined_accounting(self):
        row = ev_row()
        bad = row["request_evidence"]["tok"]["digits"]
        bad.update(error="wrong peer echo", attribution=attr(served="other-peer"), raw_prompt_tokens=-1)
        code, _, rep = run([row], CATALOG, refs_db())
        self.assertEqual(code, 0)
        identity = seller0(rep)["identity"]
        self.assertEqual(identity["evidence_state"], "unverified_attribution")
        self.assertEqual(identity["accounting"]["categories"]["negative_raw"], 1)
        self.assertEqual(identity["result"], "unknown")

    def test_transport_failure_is_not_reported_as_client_error(self):
        row = ev_row()
        row["operational_checks"] = [{"status": -1}]
        code, _, rep = run([row], CATALOG, refs_db())
        self.assertEqual(code, 0)
        self.assertEqual(seller0(rep)["availability"]["status_categories"]["transport"], 1)


class Accounting(unittest.TestCase):
    def test_negative_raw_positive_delta_recorded_not_fraud(self):
        """Negative raw usage counts are metering observations, never FABRICATED."""
        raws = {k: 9 + REF_VEC[k] for k in REF_VEC}; raws["zh"] = -7; raws["ok"] = 2
        code, _, rep = run([ev_row(raws=raws)], CATALOG, refs_db())
        s = seller0(rep)
        kinds = s["identity"]["accounting"]["categories"]
        self.assertIn("negative_raw", kinds)
        self.assertNotIn("low_count", kinds)  # negative count reported as negative_raw, not low
        # no fraud/intent verdict language in seller data (limit text explains its own policy)
        for s in rep["sellers"]:
            sj = json.dumps(s).lower()
            for word in ("fraud", "fabricated", "intent", "dishonest"):
                self.assertNotIn(word, sj)

    def test_negative_delta_positive_raws(self):
        """Reported negative delta with all-positive raws: comparison unreliable, meter follow-up."""
        deltas = {k: REF_VEC[k] for k in REF_VEC}; deltas["ru"] = -10247
        code, _, rep = run([ev_row(deltas=deltas)], CATALOG, refs_db())
        s = seller0(rep)
        self.assertIn("negative_delta", s["identity"]["accounting"]["categories"])
        self.assertNotIn("comparison_unreliable", s["identity"]["reasons"] or [])

    def test_constant_session_offsets_preserve_tokenizer_compatibility(self):
        a = ev_row(); b = ev_row()
        for k in b["tok"]: b["tok"][k] += 37
        code, _, rep = run([a, b], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(s["identity"]["result"], "compatible_tokenizer")
        self.assertNotIn("comparison_unreliable", s["identity"]["reasons"])
        self.assertIn("variable_overhead", s["identity"]["accounting"]["categories"])

    def test_fixed_low_counts_not_fraud(self):
        """Fixed identical tiny counts are metering observations, not fraud."""
        raws = {k: 1 for k in ("ok", *REF_VEC)}
        code, _, rep = run([ev_row(raws=raws)], CATALOG, refs_db())
        s = seller0(rep)
        self.assertIn("low_count", s["identity"]["accounting"]["categories"])
        self.assertIn("identical_counts_across_probes", s["identity"]["accounting"]["categories"])
        self.assertEqual(s["identity"]["result"], "inconsistent_tokenizer")  # vs exact ref — that IS allowed


class AttributionGating(unittest.TestCase):
    def test_missing_attribution_blocks_identity(self):
        row = ev_row(); [c.pop("attribution") for c in row["request_evidence"]["tok"].values()]
        code, _, rep = run([row], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(s["identity"]["result"], "unknown")
        self.assertEqual(s["identity"]["evidence_state"], "unverified_attribution")

    def test_served_peer_mismatch_blocks_identity(self):
        row = ev_row();
        for c in row["request_evidence"]["tok"].values(): c["attribution"] = attr(served="b" * 40)
        code, _, rep = run([row], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(s["identity"]["evidence_state"], "unverified_attribution")
        self.assertEqual(s["identity"]["result"], "unknown")
        kinds = {o["kind"] for o in s["identity"]["observations"]}
        self.assertIn("attribution_mismatch", kinds)

    def test_failed_cell_blocks_comparison(self):
        row = ev_row(); row["request_evidence"]["tok"]["zh"] = cell(502, attr(), error="upstream")
        code, _, rep = run([row], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(s["identity"]["evidence_state"], "cell_request_failed")

    def test_missing_cells_block_comparison(self):
        row = ev_row(); del row["request_evidence"]["tok"]["rare"]
        code, _, rep = run([row], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(s["identity"]["evidence_state"], "cells_missing")

    def test_repeats_one_compatible_one_blocked_stay_unknown(self):
        """Good repeat + blocked repeat: never pick the best one silently."""
        blocked = ev_row(); [c.pop("attribution") for c in blocked["request_evidence"]["tok"].values()]
        code, _, rep = run([ev_row(), blocked], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(s["identity"]["result"], "unknown")
        self.assertIn("unverified_attribution", s["identity"]["reasons"])

    def test_repeats_contradictory_outcomes_stay_unknown(self):
        other = ev_row()
        for k in other["tok"]:
            if k != "ok": other["tok"][k] += 3
        code, _, rep = run([ev_row(), other], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(s["identity"]["result"], "unknown")
        obs = {o["kind"] for o in s["identity"]["observations"]}
        self.assertIn("contradictory_repeats", obs)


class Operational(unittest.TestCase):
    def check(self, checks, extra={}, model=MODEL):
        row = ev_row(); row["operational_checks"] = checks; row.update(extra)
        code, _, rep = run([row], CATALOG, refs_db())
        return seller0(rep)["operational"]["recommendation"]

    def oc(self, passed, status=200, max_tokens=300, task="arithmetic", peer=PEER):
        c = {"task": task, "status": status, "expected": "42", "max_tokens": max_tokens}
        if passed is not None: c["passed"] = passed
        if peer: c["attribution"] = attr(peer)
        return c

    def test_repeats_mixed_pass_fail_not_clean_usable(self):
        """Two passes plus a failed check: repeats disagree -> unknown, never clean usable_in_test."""
        checks = [self.oc(True, task="echo"), self.oc(True, task="echo"),
                  self.oc(False, task="echo", max_tokens=300)]
        self.assertEqual(self.check(checks), "unknown")

    def test_unfinished_response_inconclusive(self):
        """passed=None with known usable budget: inconclusive, NOT failed_content."""
        self.assertNotEqual(self.check([self.oc(None, task="echo")] * 3), "avoid_for_now")

    def test_missing_budget_never_demonstrates_failure(self):
        """max_tokens absent (unknown budget): failed check stays inconclusive."""
        self.assertNotEqual(self.check([self.oc(False, max_tokens=None, task="echo")] * 3), "avoid_for_now")

    def test_operational_attribution_binds_to_row_peer(self):
        """Check attributed to an arbitrary other peer is not verified evidence."""
        row = ev_row()
        row["operational_checks"] = [self.oc(True, task="echo"), self.oc(True, task="echo")]
        for c in row["operational_checks"]: c["attribution"] = attr(peer="c" * 40, served="d" * 40)
        code, _, rep = run([row], CATALOG, refs_db())
        self.assertEqual(seller0(rep)["operational"]["recommendation"], "unknown")

    def test_passes_with_inconclusive_repeats_are_not_clean_usable(self):
        self.assertEqual(self.check([self.oc(True), self.oc(True), self.oc(None)]), "unknown")

    def test_passes_with_account_block_are_not_clean_usable(self):
        self.assertEqual(self.check([self.oc(True), self.oc(True), self.oc(None, status=403)]), "unknown")

    def test_usable_in_test(self):
        self.assertEqual(self.check([self.oc(True), self.oc(True)]), "usable_in_test")

    def test_unattributed_checks_never_usable(self):
        c1, c2 = self.oc(True), self.oc(True)
        for c in (c1, c2): del c["attribution"]
        self.assertEqual(self.check([c1, c2]), "unknown")

    def test_16token_empty_not_avoid(self):
        checks = [self.oc(False, max_tokens=16), self.oc(False, max_tokens=16)]
        self.assertEqual(self.check(checks), "unknown")

    def test_repeat_big_budget_failures_avoid(self):
        checks = [self.oc(False, task="echo", max_tokens=300), self.oc(False, task="echo", max_tokens=300),
                  self.oc(False, task="echo", max_tokens=300)]
        self.assertEqual(self.check(checks), "avoid_for_now")
        row = ev_row(); row["operational_checks"] = checks
        code, _, rep = run([row], CATALOG, refs_db())
        self.assertEqual(seller0(rep)["recommendation"]["label"], "avoid")
        self.assertIn("operational", seller0(rep)["recommendation"]["scope"])

    def test_mixed_pass_fail_not_avoid(self):
        checks = [self.oc(False, task="echo"), self.oc(False, task="echo"), self.oc(True, task="echo")]
        self.assertNotEqual(self.check(checks), "avoid_for_now")

    def test_failed_checks_with_inconclusive_repeat_cannot_avoid(self):
        for extra in (self.oc(None), {"status": 503}):
            with self.subTest(extra=extra):
                self.assertEqual(self.check([self.oc(False), self.oc(False), extra]), "unknown")

    def test_policy_auth_stays_unknown(self):
        self.assertEqual(self.check([self.oc(False, status=402)] * 4), "unknown")

    def test_seller_5xx_repeated_avoids(self):
        self.assertEqual(self.check([self.oc(False, status=503, task="echo")]), "unknown")  # only 1
        self.assertEqual(self.check([self.oc(False, status=503, task="echo"),
                                     self.oc(False, status=503, task="echo")]), "avoid_for_now")

    def test_rate_limit_stays_unknown(self):
        self.assertEqual(self.check([self.oc(False, status=429, task="echo")]), "unknown")


class CatalogCoverage(unittest.TestCase):
    def test_same_peer_two_offers_stay_separate(self):
        """Two offers sharing a peerId (different provider/serviceId) must never merge rows."""
        cat = {MODEL: {"peers": [
            {"peerId": PEER, "displayName": "Offer A", "provider": "provA", "serviceId": "svcA"},
            {"peerId": PEER, "displayName": "Offer B", "provider": "provB", "serviceId": "svcB"}]}}
        ra, rb = ev_row(), ev_row()
        ra["provider"], ra["serviceId"] = "provA", "svcA"
        rb["provider"], rb["serviceId"] = "provB", "svcB"
        code, _, rep = run([ra, rb], cat, refs_db())
        self.assertEqual(len(rep["sellers"]), 2)
        self.assertTrue(all(s["evidence_counts"]["rows"] == 1 for s in rep["sellers"]), rep["sellers"][0]["evidence_counts"])

    def test_legacy_ambiguous_binding_not_merged(self):
        """Legacy row without provider/service + two offers for same peer: no contamination."""
        cat = {MODEL: {"peers": [
            {"peerId": PEER, "displayName": "Offer A", "provider": "provA", "serviceId": "svcA"},
            {"peerId": PEER, "displayName": "Offer B", "provider": "provB", "serviceId": "svcB"}]}}
        legacy = {"model": MODEL, "peerId": PEER, "echo_ok": True, "echo_status": 200}
        code, _, rep = run([legacy], cat, refs_db())
        self.assertEqual(code, 0)
        self.assertEqual(len(rep["sellers"]), 3)  # 2 offers + 1 rows_only ambiguous bucket
        for s in rep["sellers"]:
            if s["source"] == "catalog":
                self.assertEqual(s["evidence_counts"]["rows"], 0)
                self.assertEqual(s["identity"]["result"], "unknown")

    def test_legacy_unique_binding_merges(self):
        """Legacy row without provider/service + exactly one offer for that peer: binds."""
        legacy = {"model": MODEL, "peerId": PEER, "echo_ok": True, "echo_status": 200}
        code, _, rep = run([legacy], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(s["source"], "catalog")
        self.assertEqual(s["evidence_counts"]["rows"], 1)

    def test_unrequested_seller_represented(self):
        code, _, rep = run([], CATALOG, refs_db())
        self.assertEqual(code, 0)
        s = seller0(rep)
        self.assertEqual(s["source"], "catalog")
        self.assertEqual(s["evidence_counts"]["rows"], 0)
        self.assertEqual(s["recommendation"]["label"], "unknown")

    def test_rows_only_seller_represented(self):
        code, _, rep = run([ev_row()], {}, refs_db())
        self.assertEqual(code, 0)
        self.assertEqual(len(rep["sellers"]), 1)
        self.assertEqual(seller0(rep)["source"], "rows_only")

    def test_malformed_line_rejected(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        rp = os.path.join(d, "rows.jsonl")
        Path(rp).write_text(json.dumps({"model": MODEL, "peerId": PEER}) + "\n{not json\n", encoding="utf-8")
        cp, fp = os.path.join(d, "c.json"), os.path.join(d, "r.json")
        Path(cp).write_text(json.dumps(CATALOG), encoding="utf-8")
        Path(fp).write_text(json.dumps(refs_db()), encoding="utf-8")
        p = subprocess.run([sys.executable, TOOL, rp, "--catalog", cp, "--refs", fp, "--out", os.path.join(d, "o.json")],
                           capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)

    def test_missing_peerid_rejected(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        rp = os.path.join(d, "rows.jsonl")
        Path(rp).write_text(json.dumps({"model": MODEL}) + "\n", encoding="utf-8")
        cp, fp = os.path.join(d, "c.json"), os.path.join(d, "r.json")
        Path(cp).write_text(json.dumps(CATALOG), encoding="utf-8")
        Path(fp).write_text(json.dumps(refs_db()), encoding="utf-8")
        p = subprocess.run([sys.executable, TOOL, rp, "--catalog", cp, "--refs", fp, "--out", os.path.join(d, "o.json")],
                           capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)


class OutputSafetyAndMarkdown(unittest.TestCase):
    def test_public_safety(self):
        """No raw headers/error bodies/URLs/keys in output; local row indexes preserved."""
        row = ev_row()
        row["error"] = {"error": {"message": "leak https://secret.example.com?token=abc sk-ant-key DEADBEEF",
                                  "headers": {"x-internal": "yes"}}}
        code, _, rep = run([row], CATALOG, refs_db())
        blob = json.dumps(rep)
        self.assertNotIn("secret.example.com", blob); self.assertNotIn("sk-ant-key", blob)
        self.assertNotIn("DEADBEEF", blob); self.assertNotIn("x-internal", blob)
        self.assertEqual(seller0(rep)["row_indexes"], [0])

    def test_markdown_escaping_and_existence(self):
        bad = "|pipe| `tick`le **bold**\nnewline"
        cat = {MODEL: {"peers": [{"peerId": PEER, "displayName": bad, "provider": "openai", "serviceId": MODEL}]}}
        d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, d)
        md_path = os.path.join(d, "r.md")
        rp = os.path.join(d, "rows.jsonl")
        Path(rp).write_text(json.dumps(ev_row()) + "\n", encoding="utf-8")
        cp, fp = os.path.join(d, "c.json"), os.path.join(d, "r.json")
        Path(cp).write_text(json.dumps(cat), encoding="utf-8")
        Path(fp).write_text(json.dumps(refs_db()), encoding="utf-8")
        out = os.path.join(d, "o.json")
        p = subprocess.run([sys.executable, TOOL, rp, "--catalog", cp, "--refs", fp, "--out", out, "--markdown", md_path],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        md = Path(md_path).read_text(encoding="utf-8")
        self.assertIn("\\|pipe\\|", md)
        self.assertIn("\\`tick\\`le \\*\\*bold\\*\\*", md)
        self.assertNotIn("\nnewline", md.replace("\\n", ""))
        # escaped, no live inline code / bold from input text
        self.assertNotIn("`tick`", md.replace("\\`tick\\`", ""))
        self.assertNotIn("**bold**", md.replace("\\*\\*bold\\*\\*", ""))

    def test_legacy_labels_superseded_only(self):
        row = ev_row(); row["verdict"] = "HONEST"; row["old_label"] = "ACCOUNTING-FABRICATED"
        code, _, rep = run([row], CATALOG, refs_db())
        s = seller0(rep)
        self.assertEqual(set(s["superseded"]["legacy_labels"]), {"HONEST", "ACCOUNTING-FABRICATED"})
        self.assertEqual(s["recommendation"]["label"], "unknown")
        # new recommendation must not reuse the historical label


if __name__ == "__main__":
    unittest.main()