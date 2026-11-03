"""Offline tests for the Jev gate (no API key or network needed).

    python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from gated_pipeline import GatedExtractor  # noqa: E402
from jev_gate import QUESTION_ID, JevGate, should_extract, truncate_tail  # noqa: E402
from sweep_thresholds import pick_threshold, sweep  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses and records each request body."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, json, headers, timeout):
        self.requests.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)


def ok(score, usage):
    return FakeResponse(200, {
        "model": "typesafe/jev-1.14-20261001",
        "answers": {QUESTION_ID: {"type": "noul", "noul": score}},
        "usage": usage,
    })


class FakeExtractor:
    def __init__(self):
        self.registry = []
        self.seen = []

    def observe(self, user_turn, prev_assistant_turn):
        self.seen.append(user_turn)
        self.registry.append({"text": user_turn})


def make_gate(responses, **kwargs):
    session = FakeSession(responses)
    gate = JevGate(api_key="test-key", base_url="https://example.test/api/", session=session, **kwargs)
    return gate, session


class JevGateTest(unittest.TestCase):
    def test_request_shape(self):
        gate, session = make_gate([ok(0.9, {"input_tokens": 100, "output_tokens": 5, "cost": 0.00001})],
                                  model="typesafe/jev-1.14", max_prev_chars=10)
        gate.score("From now on, answer in French.", "x" * 50)
        req = session.requests[0]
        self.assertEqual(req["url"], "https://example.test/api/v1/systemone")
        self.assertEqual(req["headers"]["Authorization"], "Bearer test-key")
        body = req["json"]
        self.assertEqual(body["model"], "typesafe/jev-1.14")
        self.assertEqual(body["questions"][QUESTION_ID]["type"], "noul")
        self.assertEqual(body["state"]["user_message"], "From now on, answer in French.")
        self.assertEqual(body["state"]["previous_assistant_message"], "..." + "x" * 10)

    def test_parses_score_and_provider_cost(self):
        gate, _ = make_gate([ok(0.83, {"input_tokens": 275, "output_tokens": 20, "cost": 0.00003})])
        result = gate.score("hi")
        self.assertAlmostEqual(result.score, 0.83)
        self.assertEqual(result.input_tokens, 275)
        self.assertEqual(result.cost_usd, 0.00003)
        self.assertEqual(result.cost_source, "provider")
        self.assertEqual(result.model, "typesafe/jev-1.14-20261001")

    def test_list_price_when_cost_missing(self):
        gate, _ = make_gate([ok(0.1, {"input_tokens": 1_000_000, "output_tokens": 12})])
        result = gate.score("hi")
        self.assertAlmostEqual(result.cost_usd, 0.042)
        self.assertEqual(result.cost_source, "list_price")

    def test_retries_on_429_then_fails_on_400(self):
        gate, session = make_gate([FakeResponse(429, {}), ok(0.5, {"input_tokens": 1, "output_tokens": 1})],
                                  max_retries=1)
        gate.score("hi")
        self.assertEqual(len(session.requests), 2)

        gate, _ = make_gate([FakeResponse(400, {"detail": "bad"})])
        with self.assertRaises(RuntimeError):
            gate.score("hi")

    def test_threshold_and_truncation(self):
        self.assertTrue(should_extract(0.5, 0.5))
        self.assertFalse(should_extract(0.49, 0.5))
        self.assertIsNone(truncate_tail(None, 5))
        self.assertEqual(truncate_tail("short", 10), "short")


class GatedExtractorTest(unittest.TestCase):
    def test_only_forwards_turns_above_threshold(self):
        usage = {"input_tokens": 10, "output_tokens": 1, "cost": 0.0}
        gate, _ = make_gate([ok(0.95, usage), ok(0.02, usage)])
        extractor = FakeExtractor()
        pipeline = GatedExtractor(gate=gate, extractor=extractor, threshold=0.3)
        self.assertTrue(pipeline.observe("From now on, use bullet points.", None))
        self.assertFalse(pipeline.observe("What is machine learning?", "..."))
        self.assertEqual(extractor.seen, ["From now on, use bullet points."])
        self.assertEqual(len(pipeline.registry), 1)
        self.assertEqual([e["extracted"] for e in pipeline.log], [True, False])
        self.assertIsNone(pipeline.log[1]["extractor_latency_s"])


class SweepTest(unittest.TestCase):
    def test_metrics_and_threshold_choice(self):
        records = [
            {"score": 0.9, "is_persistent_constraint": True},
            {"score": 0.4, "is_persistent_constraint": True},
            {"score": 0.6, "is_persistent_constraint": False},
            {"score": 0.1, "is_persistent_constraint": False},
        ]
        rows = sweep(records, [0.3, 0.5])
        low, high = rows
        self.assertEqual((low["tp"], low["fp"], low["fn"], low["tn"]), (2, 1, 0, 1))
        self.assertEqual(low["recall"], 1.0)
        self.assertEqual(low["call_reduction"], 0.25)
        self.assertEqual(high["recall"], 0.5)
        self.assertEqual(pick_threshold(rows, 0.95)["threshold"], 0.3)
        self.assertIsNone(pick_threshold(rows, 1.1))


if __name__ == "__main__":
    unittest.main()
