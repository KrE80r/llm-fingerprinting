import os
import unittest
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
if __name__ == "__main__":
    unittest.main()
