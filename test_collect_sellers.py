import unittest

from collect_sellers import check_result, select_models


class CatalogSelectionTest(unittest.TestCase):
    @staticmethod
    def offer(service="claude-opus-5-5", provider="openai"):
        return {"peerId": "seller-a", "provider": provider, "serviceId": service, "displayName": "Seller"}

    def test_exact_alias_selects_claim_not_fast_sibling(self):
        ordinary, fast = self.offer(), self.offer("claude-opus-5-5-fast")
        catalog = {"data": [
            {"id": "claude-opus-5-5-fast", "aliases": ["claude-opus-5.5-fast"], "peers": [fast]},
            {"id": "claude-opus-5-5", "aliases": ["claude-opus-5.5"], "peers": [ordinary]},
        ]}
        result = select_models(catalog, ["claude-opus-5.5"])
        self.assertEqual(result["claude-opus-5.5"]["peers"], [ordinary])
        self.assertEqual(result["claude-opus-5.5"]["catalog_id"], "claude-opus-5-5")

    def test_same_peer_different_service_is_not_lost(self):
        first, second = self.offer(), self.offer("opus-5.5", "claude-oauth")
        catalog = {"data": [{"id": "claude-opus-5.5", "peers": [first, first, second]}]}
        offers = select_models(catalog, ["claude-opus-5.5"])["claude-opus-5.5"]["peers"]
        self.assertEqual(offers, [first, second])

    def test_missing_exact_model_does_not_use_available_sibling(self):
        catalog = {"data": [{"id": "kimi-k3", "peers": [self.offer("kimi-k3")]}]}
        result = select_models(catalog, ["kimi-k3.1-pro"])
        self.assertTrue(result["kimi-k3.1-pro"]["missing"])
        self.assertEqual(result["kimi-k3.1-pro"]["peers"], [])

    def test_ambiguous_alias_cannot_choose_arbitrary_model(self):
        catalog = {"data": [
            {"id": "a", "aliases": ["claim"], "peers": []},
            {"id": "b", "aliases": ["claim"], "peers": []},
        ]}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            select_models(catalog, ["claim"])

    def test_bad_offer_is_rejected_before_requests(self):
        catalog = {"data": [{"id": "claim", "peers": [{"peerId": "seller"}]}]}
        with self.assertRaisesRegex(ValueError, "invalid peer"):
            select_models(catalog, ["claim"])


class ReadinessOutcomeTest(unittest.TestCase):
    @staticmethod
    def response(content="abc", status=200, verified=True, finish="stop"):
        return {"status": status, "attribution": {"verified": verified},
                "json": {"choices": [{"finish_reason": finish, "message": {"content": content}}]}}

    def test_wrong_or_missing_attribution_is_not_a_success(self):
        self.assertIsNone(check_result(self.response(verified=False), "abc"))
        record = self.response()
        del record["attribution"]
        self.assertIsNone(check_result(record, "abc"))

    def test_truncated_thinking_is_unknown_not_failed_task(self):
        self.assertIsNone(check_result(self.response("", finish="length"), "abc"))

    def test_completed_wrong_answer_is_a_failed_readiness_task(self):
        self.assertFalse(check_result(self.response("abd"), "abc"))
        self.assertTrue(check_result(self.response(" abc\n"), "abc"))

    def test_http_error_cannot_pass_even_with_a_cached_choice(self):
        self.assertIsNone(check_result(self.response(status=503), "abc"))

    def test_malformed_choices_are_unknown(self):
        record = self.response()
        record["json"]["choices"] = {"0": {"message": {"content": "abc"}}}
        self.assertIsNone(check_result(record, "abc"))


if __name__ == "__main__":
    unittest.main()
