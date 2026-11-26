import io
import json
import unittest

from sc_baseline.ollama_runtime import DIGEST, MODEL, VERSION, OllamaRuntime
from sc_baseline.runtime import RuntimeFailure


class FakeOllama:
    def __init__(self):
        self.model = {"name": MODEL, "digest": DIGEST}
        self.reply = {"model": MODEL, "done": True, "done_reason": "stop",
                      "message": {"content": '{"scs":[]}'}, "prompt_eval_count": 20,
                      "eval_count": 5, "load_duration": 100, "total_duration": 200,
                      "prompt_eval_duration": 40, "eval_duration": 60}
        self.generations = []

    def open(self, request, timeout):
        if request.selector == "/api/version":
            value = {"version": VERSION}
        elif request.selector == "/api/tags":
            value = {"models": [self.model]}
        else:
            self.generations.append(json.loads(request.data))
            value = self.reply
        return io.BytesIO(json.dumps(value).encode())


class OllamaTests(unittest.TestCase):
    def test_one_local_generation_and_telemetry(self):
        server = FakeOllama()
        runtime = OllamaRuntime(opener=server)
        messages = [{"role": "user", "content": "Hello"}]
        response = runtime.generate(messages)
        payload, = server.generations
        self.assertEqual((payload["messages"], payload["model"]), (messages, MODEL))
        self.assertFalse(any(payload[k] for k in ("think", "truncate", "shift", "stream")))
        self.assertEqual(payload["options"]["num_predict"], 4096)
        self.assertEqual((payload["options"]["temperature"], payload["options"]["top_p"]), (0, 1))
        self.assertEqual((response.usage["eval_count"], response.cost_usd), (5, 0.0))

    def test_digest_mismatch_prevents_generation(self):
        server = FakeOllama()
        server.model["digest"] = "different"
        with self.assertRaises(ValueError):
            OllamaRuntime(opener=server)
        self.assertFalse(server.generations)

    def test_invalid_or_thinking_envelopes_fail_once(self):
        for update in ({"message": {"content": "x", "thinking": "y"}},
                       {"done": False}, {"load_duration": "unknown"}):
            with self.subTest(update=update):
                server = FakeOllama()
                server.reply.update(update)
                with self.assertRaises(RuntimeFailure):
                    OllamaRuntime(opener=server).generate([])
                self.assertEqual(len(server.generations), 1)
