import contextlib, http.server, json, os, tempfile, threading
import unittest
from unittest.mock import patch
import fpx


TOK = {k: 0 for k in fpx.PROBES}


def ref(lab, family, tok=None):
    return {"lab": lab, "family": family, "probes": [{"tok": tok or TOK, "tok_s": None, "art_lab": None,
                                                          "reason_tok": None, "stack": {}}]}


def features(tok=None, art_lab=None):
    return {"tok": tok or TOK, "tok_s": None, "art_lab": art_lab, "reason_tok": None, "stack": {}}


class EndpointTest(unittest.TestCase):
    def test_accepts_base_or_complete_chat_url(self):
        os.environ["FPX_TEST_KEY"] = "secret"
        self.addCleanup(os.environ.pop, "FPX_TEST_KEY", None)

        self.assertEqual(fpx.endpoint("https://example.test/v1", "FPX_TEST_KEY"),
                         ("https://example.test/v1/chat/completions", "secret"))
        self.assertEqual(fpx.endpoint("https://example.test/custom/chat/completions/", "FPX_TEST_KEY"),
                         ("https://example.test/custom/chat/completions", "secret"))


class OpenSetClassificationTest(unittest.TestCase):
    def test_recognized_artifact_reports_family_without_misnaming_model(self):
        refs = {"anthropic/claude-x": ref("anthropic", "claude")}

        result = fpx.classify(features(art_lab="openai"), refs)

        self.assertEqual(result["lab"], "openai")
        self.assertEqual(result["family"], "gpt")
        self.assertEqual(result["model"], "unknown")

    def test_far_probe_abstains_instead_of_forcing_nearest_family(self):
        refs = {"anthropic/claude-x": ref("anthropic", "claude")}
        far = {k: 10 for k in fpx.PROBES}

        result = fpx.classify(features(tok=far), refs)

        self.assertEqual(result["lab"], "unknown")
        self.assertEqual(result["family"], "unknown")
        self.assertEqual(result["model"], "unknown")

    def test_ambiguous_nearest_neighbor_is_only_a_candidate(self):
        refs = {
            "anthropic/claude-a": ref("anthropic", "claude"),
            "anthropic/claude-b": ref("anthropic", "claude"),
        }

        result = fpx.classify(features(), refs)

        self.assertEqual(result["lab"], "anthropic")
        self.assertEqual(result["family"], "claude")
        self.assertEqual(result["model"], "unknown")
        self.assertIn(result["candidate_model"], refs)


class LeaveOneLabOutTest(unittest.TestCase):
    def test_counts_correct_unknown_and_wrong_family_names(self):
        one = {k: 1 for k in fpx.PROBES}
        far = {k: 10 for k in fpx.PROBES}
        refs = {
            "anthropic/claude-a": ref("anthropic", "claude"),
            "openai/gpt-a": ref("openai", "gpt", one),
        }

        outcomes, _, abstained = fpx.leave_one_lab_out(refs, [
            ("anthropic", features(art_lab="anthropic")),
            ("anthropic", features(tok=far)),
            ("anthropic", features(tok=one)),
        ])

        self.assertEqual(outcomes, {"correct": 1, "unknown": 1, "wrong": 1})
        self.assertEqual(abstained, 0)


class ReferenceDatabaseTest(unittest.TestCase):
    def test_database_records_protocol_and_abstention_bounds(self):
        database = fpx.make_reference_db({"anthropic/claude-a": ref("anthropic", "claude")})

        self.assertEqual(database["schema"], "fpx.reference-db/v1")
        self.assertEqual(database["protocol"]["id"], "fpx-stack-v1")
        self.assertEqual(database["protocol"]["tokenizer_probes"], fpx.PROBES)
        self.assertEqual(database["calibration"]["max_family_distance"], fpx.MAX_FAMILY_DISTANCE)
        self.assertEqual(database["calibration"]["min_structural_lead"], fpx.MIN_STRUCTURAL_LEAD)
        self.assertIn("anthropic/claude-a", database["models"])

    def test_unusable_reference_requires_token_counts(self):
        missing = {k: None for k in fpx.PROBES}

        self.assertFalse(fpx.usable_ref(ref("deepseek", "deepseek", missing)))


class EvidenceFusionTest(unittest.TestCase):
    def test_single_contradiction_abstains(self):
        self.assertEqual(fpx.fuse_results(["PASS", "PASS", "PASS", "FAIL"]), "INDETERMINATE")

    def test_three_noncontradictory_groups_are_consistent(self):
        self.assertEqual(fpx.fuse_results(["PASS", "PASS", "PASS", "ABSTAIN"]), "CONSISTENT_WITH")

    def test_two_contradictions_are_reported(self):
        self.assertEqual(fpx.fuse_results(["PASS", "PASS", "FAIL", "FAIL"]), "CONTRADICTED")

    def test_ambiguous_exact_reference_is_not_a_mismatch(self):
        refs = {"anthropic/claude-a": ref("anthropic", "claude"),
                "anthropic/claude-b": ref("anthropic", "claude")}
        result = fpx.verify_ref(features(), refs, "anthropic/claude-a")
        self.assertEqual(result["verdict"], "INDETERMINATE")
        self.assertIsNone(result["checks"]["nearest_is_claimed"])

    def test_reference_incompatibility_is_still_reported(self):
        refs = {"anthropic/claude-a": ref("anthropic", "claude")}
        result = fpx.verify_ref(features(tok={k: 10 for k in fpx.PROBES}), refs, "anthropic/claude-a")
        self.assertEqual(result["verdict"], "MISMATCH")
        self.assertFalse(result["checks"]["tokenizer+stack"])

    def test_missing_exact_reference_abstains(self):
        for observation in (features(), features(tok={k: None for k in fpx.PROBES})):
            with self.subTest(observation=observation):
                self.assertEqual(fpx.verify_ref(observation, {}, "anthropic/claude-a")["verdict"],
                                 "insufficient_reference")

    def test_gpt_twins_are_group_consistency_not_exact_identity(self):
        observation = {"label": fpx.fp2.GPT_TWIN, "consistent": True,
                       "gate_label": fpx.fp2.GPT_TWIN, "habit": {}}
        with patch.object(fpx.fp2, "identify", return_value=observation):
            for claimed in fpx.fp2.GPT_TWIN.split("|"):
                with self.subTest(claimed=claimed):
                    self.assertEqual(fpx.verify_family("synthetic-capture", claimed)["verdict"],
                                     "CONSISTENT_WITH_GROUP")



class TokenIdNamespaceTest(unittest.TestCase):
    IDS = [32, 100276, 100277]
    DB = {
        "schema": "fpx.token-id-reference/v1",
        "protocol": {"token_ids": IDS},
        "namespaces": {
            "openai-cl100k": {"acceptance": [True, True, False], "echoes": ["A", "x", ""],
                              "family_candidates": ["gpt"]},
            "openai-o200k": {"acceptance": [True, True, True], "echoes": ["A", "x", "x"],
                             "family_candidates": ["gpt"]},
        },
    }

    @staticmethod
    def rows(signature, banned="B", other="x"):
        rows = []
        for repeat in ("r1", "r2"):
            rows.extend({
                "repeat": repeat, "token_id": token_id, "bias": 100,
                "status": 200 if accepted else 400,
                "json": {"choices": [{"message": {"content": "A" if token_id == 32 else other}}]},
            } for token_id, accepted in zip(TokenIdNamespaceTest.IDS, signature))
            rows.append({
                "repeat": repeat, "token_id": 32, "bias": -100, "status": 200,
                "json": {"choices": [{"message": {"content": banned}}]},
            })
        return rows

    def test_names_only_a_unique_stable_controlled_namespace(self):
        result = fpx.classify_token_id_rows(self.rows([True, True, False]), self.DB)

        self.assertEqual(result["verdict"], "CONSISTENT_WITH")
        self.assertEqual(result["namespace"], "openai-cl100k")
        self.assertEqual(result["family_candidates"], ["gpt"])

    def test_ignored_negative_bias_abstains(self):
        result = fpx.classify_token_id_rows(self.rows([True, True, False], banned="A"), self.DB)

        self.assertEqual(result["verdict"], "ABSTAIN")
        self.assertEqual(result["reason"], "control_failed")


    def test_different_token_echo_abstains(self):
        result = fpx.classify_token_id_rows(self.rows([True, True, False], other="different"), self.DB)

        self.assertEqual(result["verdict"], "ABSTAIN")
        self.assertEqual(result["reason"], "echo_mismatch")

    def test_transport_error_abstains(self):
        rows = self.rows([True, True, False])
        rows[1]["status"] = 429

        result = fpx.classify_token_id_rows(rows, self.DB)

        self.assertEqual(result["verdict"], "ABSTAIN")
        self.assertEqual(result["reason"], "transport_error")

    def test_acceptance_collision_abstains_before_echo_filtering(self):
        database = {**self.DB, "namespaces": {
            **self.DB["namespaces"],
            "collision": {"acceptance": [True, True, False], "echoes": ["A", "different", ""]},
        }}

        result = fpx.classify_token_id_rows(self.rows([True, True, False]), database)

        self.assertEqual(result["verdict"], "ABSTAIN")
        self.assertEqual(result["reason"], "reference_collision")

    def test_one_accepted_id_reference_is_underdetermined(self):
        database = {**self.DB, "namespaces": {
            "small-or-cl100k": {"acceptance": [True, False, False], "echoes": ["A", "", ""]},
        }}

        result = fpx.classify_token_id_rows(self.rows([True, False, False]), database)

        self.assertEqual(result["verdict"], "ABSTAIN")
        self.assertEqual(result["reason"], "underdetermined_reference")


# ------------------------------------------------------------------ pin/attribution integration
# Synthetic fixtures only: peer ids are fake, every response is produced by a local deterministic server.
PIN = "a" * 32
OTHER = "b" * 32


def _completion(text="ok", prompt_tokens=11, reasoning_tokens=7):
    return {"choices": [{"message": {"content": text, "reasoning_details": []}, "finish_reason": "stop",
                         "native_finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5,
                      "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
            "system_fingerprint": "fpv0_42424242"}


class _ChatServer:
    """Deterministic chat-completions double for integration tests. Pinned requests are answered with an
    x-antseed-peer-id echo of the requested peer unless the last message matches `wrong` (echoes a
    different peer) or `missing` (no echo). All peer ids and model ids here are synthetic."""
    def __init__(self, wrong=(), missing=()):
        self.wrong, self.missing, self.requests = list(wrong), list(missing), []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                last = (body.get("messages") or [{}])[-1].get("content", "")
                flat = last if isinstance(last, str) else " ".join(p.get("text", "") for p in last if isinstance(p, dict))
                status, payload = outer._answer(flat, body)
                req = self.headers.get("x-antseed-pin-peer")
                served = req if req is not None else None
                if req is not None:
                    if any(w in flat or w in json.dumps(body) for w in outer.wrong): served = OTHER
                    if any(w in flat for w in outer.missing): served = None
                outer.requests.append((flat[:40], req, served))
                self.send_response(status)
                self.send_header("content-type", "application/json")
                if served is not None: self.send_header("x-antseed-peer-id", served)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)

    def _answer(self, flat, body):
        if flat == "invalid-json-object": return 200, ["not a completion object"]
        if (body.get("response_format") or {}).get("type") == "json_schema":
            return 400, {"error": {"message": "response_format type bogus_type is not supported"}}
        if "logit_bias" in body:
            (tid_s, bias), = body["logit_bias"].items()
            tid = int(tid_s)
            if int(bias) == -100: return 200, _completion("B", 5)
            if tid == 32: return 200, _completion("A", 5)
            if tid == 100276: return 200, _completion("x", 5)
            return 400, {"error": {"message": "logit_bias token rejected"}}
        if "tools" in body:
            msg = {"content": None, "tool_calls": [{"id": "call_abcdef12", "type": "function",
                   "function": {"name": "emit", "arguments": "{\"v\": \"hello world\"}"}}]}
            return 200, {"choices": [{"message": msg, "finish_reason": "tool_calls",
                                      "native_finish_reason": "tool_calls"}],
                         "usage": {"prompt_tokens": 40, "completion_tokens": 9}}
        return 200, _completion("7", 8 + len(flat.encode()))

    @property
    def url(self): return f"http://127.0.0.1:{self.httpd.server_address[1]}"


@contextlib.contextmanager
def chat_server(wrong=(), missing=()):
    s = _ChatServer(wrong, missing)
    t = threading.Thread(target=s.httpd.serve_forever, daemon=True)
    t.start()
    try: yield s
    finally:
        s.httpd.shutdown(); s.httpd.server_close()


class PinIntegrationTest(unittest.TestCase):
    NS_IDS = [32, 100276, 100277]
    NSDB = {"schema": "fpx.token-id-reference/v1", "protocol": {"token_ids": NS_IDS},
            "namespaces": {"cl100k-synthetic": {"acceptance": [True, True, False], "echoes": ["A", "x", ""],
                                                "family_candidates": ["synth"]}}}

    def test_verified_pin_probe_collects_per_request_evidence(self):
        with chat_server() as s:
            p = fpx.probe("vendor/synth-1", s.url, None, None, pin=PIN)
        self.assertTrue(p["pin_ok"]); self.assertEqual(p["pin_peer"], PIN)
        cell = p["request_evidence"]["tok"]["ok"]
        self.assertEqual(cell["status"], 200)
        self.assertEqual(cell["attribution"], {"requested_peer": PIN, "served_peer": PIN, "verified": True})
        self.assertTrue(all(c["attribution"]["verified"] for c in p["request_evidence"]["tok"].values()))
        self.assertEqual(p["reason"]["attribution"]["verified"], True)   # pin verified per request
        self.assertIn("attribution", p["stack"]["sys"])                  # stack/reason/behav records carry it too
        f = fpx.features(p)
        self.assertIn("sys", f["stack"]); self.assertTrue(f["behav"]); self.assertEqual(f["reason_tok"], 7)
        self.assertEqual(f["tok"]["zh"], p["tok"]["zh"] - p["tok"]["ok"])

    def test_wrong_echo_quarantines_tok_cell_but_keeps_raw_observation(self):
        with chat_server(wrong=("31415926535897932384",)) as s:   # only the "digits" probe
            p = fpx.probe("vendor/synth-1", s.url, None, None, pin=PIN)
        cell = p["request_evidence"]["tok"]["digits"]
        self.assertEqual(cell["attribution"]["served_peer"], OTHER)
        self.assertIn("not " + PIN, cell["error"])
        self.assertIsNone(p["tok"]["digits"])                       # cannot become tokenizer evidence
        self.assertIsNotNone(cell["raw_prompt_tokens"])             # raw accounting kept for audit
        self.assertTrue(p["tok_raw"]["digits"]["quarantined"])
        self.assertFalse(p["pin_ok"])
        f = fpx.features(p)
        self.assertIsNone(f["tok"]["digits"])
        self.assertIsNotNone(f["tok"]["zh"])                        # verified cells stay usable

    def test_missing_echo_quarantines(self):
        with chat_server(missing=("\t\t",)) as s:                   # only the "ws" probe
            p = fpx.probe("vendor/synth-1", s.url, None, None, pin=PIN)
        cell = p["request_evidence"]["tok"]["ws"]
        self.assertEqual(cell["attribution"]["served_peer"], None)
        self.assertEqual(cell["error"], "no x-antseed-peer-id header on response")
        self.assertIsNone(p["tok"]["ws"])
        self.assertFalse(p["pin_ok"])

    def test_wrong_echo_inside_behaviour_battery_gates_whole_behaviour_block(self):
        with chat_server(wrong=("strawberry raspberry",)) as s:
            p = fpx.probe("vendor/synth-1", s.url, None, None, pin=PIN)
        self.assertTrue(p["behav"]["count"]["quarantined"])
        self.assertEqual(p["behav"]["count"]["attribution"]["served_peer"], OTHER)
        self.assertEqual(fpx.behav_sig(p), {})                      # battery is one block: no mixing
        self.assertIn("sys", fpx.stack_sig(p))                      # verified structural block unaffected
        self.assertFalse(p["pin_ok"])

    def test_reason_attribution_gates_artifact_and_behaviour(self):
        with chat_server(wrong=("391 prime",)) as s:                # only the dedicated reasoning request
            p = fpx.probe("vendor/synth-1", s.url, None, None, pin=PIN)
        self.assertTrue(p["reason"]["quarantined"])
        self.assertFalse(p["reason"]["attribution"]["verified"])
        f = fpx.features(p)
        self.assertIsNone(f["art_lab"]); self.assertIsNone(f["reason_tok"])
        self.assertEqual(f["behav"], {}); self.assertEqual(f["stack"], {})
        self.assertFalse(p["pin_ok"])

    def test_wrong_peer_error_cannot_become_a_capability_fingerprint(self):
        with chat_server(wrong=("bogus_type",)) as s:
            p = fpx.probe("vendor/synth-1", s.url, pin=PIN)
        self.assertEqual(p["stack"]["err"]["status"], 400)
        self.assertTrue(p["stack"]["err"]["quarantined"])
        self.assertFalse(p["pin_ok"])
        self.assertEqual(fpx.stack_sig(p), {})

    def test_nonobject_response_is_retained_but_not_evidence(self):
        with chat_server() as s:
            row = fpx.gate(fpx._chat(s.url + "/chat/completions", "", "vendor/synth-1",
                           [{"role": "user", "content": "invalid-json-object"}], pin=PIN))
        self.assertEqual(row["raw_json"], ["not a completion object"])
        self.assertNotIn("json", row)
        self.assertTrue(row["quarantined"])

    def test_http_failed_battery_request_is_neither_quarantined_nor_evidence(self):
        with chat_server() as s:
            p = fpx.probe("vendor/synth-1", s.url, None, None, pin=PIN)
        err = p["stack"]["err"]                                     # BAD_SCHEMA -> 400 with peer echo
        self.assertEqual(err["status"], 400)
        self.assertFalse(err["quarantined"])                        # 400 with peer echo: failed, not fake
        self.assertEqual(fpx.stack_sig(p)["err"].split(":")[0], "400")
        with chat_server() as s:
            rec = fpx._chat(s.url + "/chat/completions", "", "vendor/synth-1",
                            [{"role": "user", "content": "ok"}], extra=fpx.BAD_SCHEMA, pin=PIN)
            self.assertEqual(rec["status"], 400)
            self.assertTrue(rec["headers"])                         # error responses keep headers
            self.assertEqual(rec["attribution"]["verified"], True)
            self.assertTrue(rec["error"])

    def test_unpinned_probe_is_unchanged(self):
        with chat_server() as s:
            p = fpx.probe("vendor/synth-1", s.url)
        self.assertIsNone(p["pin_peer"]); self.assertIsNone(p["pin_ok"])
        self.assertFalse(any("attribution" in json.dumps(v) for v in p["stack"].values()))
        self.assertEqual(sorted(p["tok"]), sorted(["ok", *fpx.PROBES]))
        self.assertTrue(all(v is not None for v in p["tok"].values()))
        self.assertTrue(all(c["attribution"] is None for c in p["request_evidence"]["tok"].values()))
        f = fpx.features(p)
        self.assertIn("sys", f["stack"]); self.assertTrue(f["behav"])

    def test_token_id_namespace_pinned_verified_and_abstains_when_unverified(self):
        dbp = tempfile.mktemp()
        with open(dbp, "w") as fh: json.dump(self.NSDB, fh)
        self.addCleanup(os.unlink, dbp)
        with chat_server() as s:
            r = fpx.identify_token_id_namespace(dbp, "vendor/synth-1", s.url, None, None, pin=PIN)
        self.assertEqual(r["result"]["verdict"], "CONSISTENT_WITH")
        self.assertEqual(r["pin_peer"], PIN); self.assertTrue(r["pin_ok"])
        with chat_server(missing=("x",)) as s:                      # every row unanswered -> no echo
            r = fpx.identify_token_id_namespace(dbp, "vendor/synth-1", s.url, None, None, pin=PIN)
        self.assertEqual(r["result"], {"verdict": "ABSTAIN", "reason": "pin_unverified", "namespace": "unknown"})
        self.assertFalse(r["pin_ok"])
        self.assertFalse(any(o.get("quarantined") and o.get("attribution", {}).get("verified")
                             for o in r["observations"]))
        with chat_server() as s:                                    # untargeted callers unchanged
            r = fpx.identify_token_id_namespace(dbp, "vendor/synth-1", s.url)
        self.assertEqual(r["result"]["verdict"], "CONSISTENT_WITH"); self.assertIsNone(r["pin_ok"])

    def test_distribution_rows_carry_attribution_and_invalid_rows_are_excluded(self):
        with chat_server(wrong=("a random number between 1 and 100",)) as s:
            rows = fpx.sample_rows("vendor/synth-1", "t", 1, s.url, None, None, pin=PIN)
        self.assertTrue(all("attribution" in r for r in rows))
        bad = [r for r in rows if r["task"] == "number_1_100"]
        self.assertTrue(bad and all(r["attribution"]["served_peer"] == OTHER for r in bad))
        self.assertTrue(all(r["text"] for r in bad))                # raw observations kept
        grouped = fpx.group_samples(rows)
        self.assertNotIn("number_1_100", next(iter(grouped.values())))
        self.assertIn("letter", next(iter(grouped.values())))
        with chat_server() as s:
            rows = fpx.sample_rows("vendor/synth-1", "t", 1, s.url)  # unpinned compatibility
        self.assertTrue(all(r.get("attribution") is None and "attribution_error" not in r for r in rows))
        self.assertIn("number_1_100", next(iter(fpx.group_samples(rows).values())))



if __name__ == "__main__":
    unittest.main()
