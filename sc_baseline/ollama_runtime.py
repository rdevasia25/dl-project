"""Local, pinned Qwen runtime for the shared extractor."""
import hashlib
import json
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .runtime import RuntimeFailure, _NoRedirect, configuration_id
from .types import ModelResponse

BASE_URL = "http://127.0.0.1:11439"
MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
VERSION = "0.40.18"
OPTIONS = {
    "num_ctx": 16384, "num_predict": 4096, "temperature": 0, "top_p": 1,
    "presence_penalty": 0, "repeat_penalty": 1,
}
CHAT_SETTINGS = {
    "stream": False, "think": False, "truncate": False, "shift": False,
    "options": OPTIONS, "keep_alive": "10m",
}


class OllamaRuntime:
    provider = "ollama"
    model = MODEL

    def __init__(self, *, opener=None):
        self._opener = opener or build_opener(ProxyHandler({}), _NoRedirect())
        try:
            version = self._json("/api/version").get("version")
            models = self._json("/api/tags").get("models", [])
            matching = any(isinstance(m, dict) and m.get("name") == MODEL
                           and m.get("digest") == DIGEST for m in models)
        except (RuntimeFailure, TypeError) as exc:
            raise ValueError(f"Cannot inspect local Ollama at {BASE_URL}.") from exc
        if version != VERSION or not matching:
            raise ValueError("Install the pinned Ollama version and Qwen model digest.")
        self.metadata = {"version": version, "model": MODEL, "digest": DIGEST,
                         "endpoint": BASE_URL, "settings": CHAT_SETTINGS}
        identity = [configuration_id(self.provider, MODEL), self.metadata]
        self.config_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def _json(self, path, payload=None):
        request = Request(BASE_URL + path,
                          data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=300) as response:
                data = json.load(response)
        except HTTPError as exc:
            exc.close()
            raise RuntimeFailure("http_error", f"Ollama HTTP status {exc.code}.") from None
        except (OSError, URLError):
            raise RuntimeFailure("transport_error", "Local Ollama connection failed.") from None
        except (ValueError, UnicodeError):
            raise RuntimeFailure("provider_schema", "Invalid local Ollama JSON.") from None
        if not isinstance(data, dict) or "error" in data:
            raise RuntimeFailure("provider_schema", "Invalid local Ollama envelope.")
        return data

    def generate(self, messages):
        data = self._json("/api/chat", {"model": MODEL, "messages": messages, **CHAT_SETTINGS})
        reply = data.get("message")
        if (data.get("done") is not True or not isinstance(reply, dict)
                or not isinstance(reply.get("content"), str) or not isinstance(data.get("model"), str)
                or not isinstance(data.get("done_reason"), str)):
            raise RuntimeFailure("provider_schema", "Incomplete local Ollama response.")
        if reply.get("thinking") or "<think>" in reply["content"]:
            raise RuntimeFailure("thinking_enabled", "Ollama returned thinking text.")
        usage = {}
        for key in ("prompt_eval_count", "eval_count", "load_duration", "total_duration",
                    "prompt_eval_duration", "eval_duration"):
            value = data.get(key)
            if type(value) is not int or value < 0:
                raise RuntimeFailure("provider_schema", f"Invalid Ollama telemetry field {key}.")
            usage[key] = value
        return ModelResponse(reply["content"], data["model"], usage, 0.0,
                             "local:no_provider_fee", data["done_reason"])
