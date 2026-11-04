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

    def test_estimated_token_counts_are_never_named_even_when_they_match(self):
        refs = {"anthropic/claude-x": ref("anthropic", "claude")}
        estimator = fpx.ACCT_LEN // 3   # ~3 chars/token heuristic, as served by stealth/crouton-alpha
        f = {**features(), "acct_tok": estimator, "accounting": fpx.accounting(estimator)}

        result = fpx.classify(f, refs)

        self.assertEqual(result["accounting"], "impossible")
        self.assertEqual((result["family"], result["model"]), ("unknown", "unknown"))
        self.assertNotIn("candidate_model", result)
        self.assertEqual(fpx.accounting(fpx.ACCT_LEN), "ok")   # one token per random letter is a real tokenizer


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


class OutputBoundaryTest(unittest.TestCase):
    TEXT = "abcdefghijklmnopqrstuvwxyz" * 4
    DB = {"texts": [TEXT], "tokenizers": {
        "evens": {"boundaries": [list(range(2, 104, 2))], "models": ["lab-a/x"]},
        "fours": {"boundaries": [list(range(4, 104, 4))], "models": ["lab-b/y"]},
        "triples": {"boundaries": [list(range(3, 104, 3))], "models": ["lab-c/z"]},
    }}

    def rows(self, cuts, finish="length"):
        return [{"finish": finish, "text": self.TEXT[:c]} for c in cuts]

    def classify(self, cuts, db=DB):
        return fpx.classify_cuts([fpx.cut_points(self.rows(cuts), self.TEXT)], db)

    def test_names_the_only_tokenizer_with_every_cut_as_a_boundary(self):
        result = self.classify(range(2, 60, 2))

        self.assertEqual(result["verdict"], "MATCH")
        self.assertEqual(result["tokenizer"], "evens")
        self.assertEqual(result["models"], ["lab-a/x"])

    def test_cuts_shared_by_two_tokenizers_do_not_name_either(self):
        # every cut is a multiple of 4, so 'evens' and 'fours' both fit
        result = self.classify(range(4, 100, 4))

        self.assertEqual(result["verdict"], "ABSTAIN")
        self.assertEqual(result["reason"], "weak_separation")
        self.assertEqual(set(result["consistent_with"]), {"evens", "fours"})

    def test_two_separating_cuts_are_not_enough(self):
        result = self.classify(list(range(4, 84, 4)) + [2, 6])   # only two cuts contradict 'fours'

        self.assertEqual(result["verdict"], "ABSTAIN")
        self.assertEqual(result["consistent_with"], ["evens", "fours"])

    def test_tokenizer_without_reference_on_a_text_is_never_ruled_out_there(self):
        db = {"texts": [self.TEXT, self.TEXT], "tokenizers": {
            "evens": {"boundaries": [list(range(2, 104, 2)), list(range(2, 104, 2))], "models": ["lab-a/x"]},
            "fours": {"boundaries": [list(range(4, 104, 4)), None], "models": ["lab-b/y"]},
        }}
        first = fpx.cut_points(self.rows(range(4, 100, 4)), self.TEXT)   # fits both
        second = fpx.cut_points(self.rows(range(2, 60, 2)), self.TEXT)   # would rule out 'fours', but it has no reference

        result = fpx.classify_cuts([first, second], db)

        self.assertEqual(result["verdict"], "ABSTAIN")
        self.assertEqual(result["contradictions"]["fours"], 0)

    def test_provider_recutting_by_characters_matches_nothing(self):
        # a proxy that truncates at ~chars/n instead of model tokens lands between every tokenizer's boundaries
        result = self.classify(range(5, 104, 5))

        self.assertEqual(result["verdict"], "NOT_ENROLLED")

    def test_only_truncated_exact_prefixes_count_as_cuts(self):
        rows = (self.rows([10]) + self.rows([12], finish="stop") + [{"finish": "length", "text": "abcX"}]
                + [{"finish": "length", "text": "  abcd"}, {"finish": "length", "text": "abcdef\ufffd"}])

        self.assertEqual(fpx.cut_points(rows, self.TEXT), {10, 4, 6})

    def test_word_level_release_cannot_contribute_cuts(self):
        # crouton-alpha's stealth layer released whole words late in a reply: cuts after a space, before punctuation
        # or between CJK characters must be ignored, mid-word cuts kept
        text = "Vulpes vulpes, e.g. 我们今天 unbelievably"
        rows = [{"finish": "length", "text": text[:c]} for c in (7, 13, 14, 20, 21, 30, 32)]

        self.assertEqual(fpx.cut_points(rows, text), {30, 32})


if __name__ == "__main__":
    unittest.main()
