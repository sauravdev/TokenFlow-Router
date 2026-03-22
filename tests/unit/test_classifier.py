"""Unit tests for the request classifier."""

import pytest
from tokenflow.classifier import RequestClassifier, _token_band, _count_input_tokens
from tokenflow.models import LatencyClass, PriorityTier, TokenBand, WorkloadType


clf = RequestClassifier()


def test_token_band_boundaries():
    assert _token_band(100) == TokenBand.TINY
    assert _token_band(256) == TokenBand.TINY
    assert _token_band(257) == TokenBand.SMALL
    assert _token_band(1024) == TokenBand.SMALL
    assert _token_band(1025) == TokenBand.MEDIUM
    assert _token_band(4096) == TokenBand.MEDIUM
    assert _token_band(4097) == TokenBand.LARGE
    assert _token_band(16384) == TokenBand.LARGE
    assert _token_band(16385) == TokenBand.XLARGE


def test_count_input_tokens_from_messages():
    body = {
        "model": "llama3",
        "messages": [
            {"role": "user", "content": "a" * 400},  # 400 chars = ~100 tokens
        ],
    }
    tokens = _count_input_tokens(body)
    assert 80 <= tokens <= 120


def test_prefill_heavy_classification():
    # 8000 input, 200 output → prefill heavy
    body = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "x " * 8000}],
        "max_tokens": 200,
    }
    profile = clf.classify(body)
    assert profile.workload_type == WorkloadType.PREFILL_HEAVY


def test_decode_heavy_classification():
    # 100 input, 2000 output → decode heavy
    body = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Tell me a long story."}],
        "max_tokens": 2000,
    }
    profile = clf.classify(body)
    assert profile.workload_type == WorkloadType.DECODE_HEAVY


def test_reasoning_classification():
    body = {
        "model": "nvidia/llama-3.1-nemotron-70b-reasoning",
        "messages": [{"role": "user", "content": "Solve this problem step by step."}],
        "max_tokens": 1024,
    }
    profile = clf.classify(body)
    assert profile.workload_type == WorkloadType.REASONING


def test_premium_priority_gives_interactive_latency():
    body = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
    }
    profile = clf.classify(body, priority_tier=PriorityTier.PREMIUM)
    assert profile.latency_class == LatencyClass.INTERACTIVE


def test_batch_priority_gives_batch_latency():
    body = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Classify this."}],
        "max_tokens": 16,
    }
    profile = clf.classify(body, priority_tier=PriorityTier.BATCH)
    assert profile.latency_class == LatencyClass.BATCH


def test_streaming_flag_preserved():
    body = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    profile = clf.classify(body)
    assert profile.streaming is True


def test_tenant_and_app_id_set():
    body = {"model": "llama3", "messages": [{"role": "user", "content": "x"}]}
    profile = clf.classify(body, tenant_id="acme", app_id="chat-app")
    assert profile.tenant_id == "acme"
    assert profile.app_id == "chat-app"
