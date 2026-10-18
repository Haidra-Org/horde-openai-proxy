"""Exercise the real proxy caches without starting pollers or calling services."""

import importlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cachetools import TTLCache


@pytest.fixture
def modules():
    with patch.dict(os.environ, {
        "TOKENIZERS_CONFIG_DIR": "/tmp/proxy-test-tokenizers",
        "HF_TOKEN": "",
        "HORDE_PROXY_PASSKEY": "test-only",
    }), patch(
        "requests.sessions.Session.request",
        side_effect=AssertionError("Tests must not call external services"),
    ):
        with patch.object(threading.Thread, "start"):
            workers = importlib.import_module("horde_openai_proxy.workers")
            model = importlib.import_module("horde_openai_proxy.model")
        yield workers, model


@pytest.mark.parametrize("name", [
    "get_max_context_length_for_model",
    "get_max_tokens_for_model",
    "get_references",
    "get_models",
])
def test_concurrent_cache_access_is_serialized(modules, monkeypatch, name):
    workers, model = modules
    if name.startswith("get_max_"):
        # Avoid the constructor's background worker poller.
        instance = object.__new__(workers.HordeWorkers)
        instance.workers = [SimpleNamespace(
            models=["model-a", "model-b"],
            max_context_length=8192,
            max_length=512,
        )]
        function = getattr(instance, name)
        expected = 8192 if name == "get_max_context_length_for_model" else 512
        invoke = lambda i: function("model-a" if i % 2 else "model-b")
    elif name == "get_references":
        monkeypatch.setattr(model.requests, "get", lambda *a, **kw:
                            SimpleNamespace(json=lambda: {"reference": "value"}))
        function = model.get_references
        expected = {"reference": "value"}
        invoke = lambda i: function()
    else:
        monkeypatch.setattr(model, "get_references", lambda: {})
        monkeypatch.setattr(model, "get_horde_models", lambda ip: [])
        function = model.get_models
        expected = {}
        invoke = lambda i: function("test-client-{}".format(i))

    cache = function.cache
    function.cache_clear()
    guard = threading.Lock()
    active = {}

    def checked(operation):
        def wrapper(self, *args, **kwargs):
            if self is not cache:
                return operation(self, *args, **kwargs)
            thread_id = threading.get_ident()
            with guard:
                active[thread_id] = active.get(thread_id, 0) + 1
                overlapping = len(active) > 1
            try:
                assert not overlapping, "Concurrent access to shared TTLCache"
                # Widen the race window so an unlocked decorator fails reliably.
                time.sleep(0.001)
                return operation(self, *args, **kwargs)
            finally:
                with guard:
                    active[thread_id] -= 1
                    if active[thread_id] == 0:
                        del active[thread_id]
        return wrapper

    monkeypatch.setattr(TTLCache, "__getitem__", checked(TTLCache.__getitem__))
    monkeypatch.setattr(TTLCache, "__setitem__", checked(TTLCache.__setitem__))
    start = threading.Barrier(16)

    def request(i):
        start.wait(timeout=10)
        # Multiple keys force eviction in the worker caches (maxsize=1).
        return [invoke(i + offset) for offset in range(8)]

    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(request, range(16)))
        assert results == [[expected] * 8] * 16
        assert cache.currsize <= cache.maxsize
    finally:
        function.cache_clear()


@pytest.mark.parametrize("name, fallback, populated", [
    ("get_max_context_length_for_model", 2048, 8192),
    ("get_max_tokens_for_model", 120, 512),
])
def test_worker_cache_expiry_and_fallback(modules, name, fallback, populated):
    workers, _ = modules
    instance = object.__new__(workers.HordeWorkers)
    instance.workers = []
    function = getattr(instance, name)
    function.cache_clear()
    try:
        assert function("model-a") == fallback
        instance.workers = [SimpleNamespace(
            models=["model-a"], max_context_length=8192, max_length=512,
        )]
        assert function("model-a") == fallback
        function.cache.expire(time=function.cache.timer() + 31)
        assert function("model-a") == populated
        assert function("unknown-model") == fallback
    finally:
        function.cache_clear()


def test_cache_misses_do_not_hold_lock_during_http_requests(modules, monkeypatch):
    _, model = modules
    both_fetching = threading.Barrier(2)

    def fetch(*args, **kwargs):
        # A lock held across HTTP would prevent the second request reaching here.
        both_fetching.wait(timeout=5)
        return SimpleNamespace(json=lambda: {"reference": "value"})

    monkeypatch.setattr(model.requests, "get", fetch)
    model.get_references.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda i: model.get_references(), range(2)))
        assert results == [{"reference": "value"}] * 2
    finally:
        model.get_references.cache_clear()
