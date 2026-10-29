"""Single-flight locks survive eviction and do not retain idle request keys."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import weakref
from collections import OrderedDict

import pytest

# Importing mteb.api also creates the app, so all API extras must be present.
for dependency in (
    "fastapi",
    "pydantic_settings",
    "prometheus_client",
    "opentelemetry.sdk",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.instrumentation.fastapi",
):
    pytest.importorskip(dependency, exc_type=ImportError)

from mteb.api import cache  # noqa: E402 -- optional API extras checked above


def _run(scenario):
    return asyncio.run(asyncio.wait_for(scenario, timeout=5))


async def _ready_call(ready, layer, key, build):
    ready.set()
    return await cache._cache_or_build(layer, key, build)


async def _finish_tasks(tasks):
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("capacity", [1, 256])
def test_waiting_requests_share_one_rebuild_after_lru_eviction(capacity: int):
    async def scenario():  # noqa: PLR0914 -- explicit concurrent task/event roles
        layer = cache.CacheLayer("test", OrderedDict(), max_size=capacity)
        fillers = list(range(capacity - 1))
        first_entered, first_release = asyncio.Event(), asyncio.Event()
        waiter_ready = asyncio.Event()
        b_entered, b_release = asyncio.Event(), asyncio.Event()
        rebuild_entered, rebuild_release = asyncio.Event(), asyncio.Event()
        calls = active = peak = 0
        first_value, rebuilt_value = object(), object()
        tasks = []

        async def a_build():
            nonlocal calls, active, peak
            calls += 1
            attempt = calls
            active += 1
            peak = max(peak, active)
            try:
                if attempt == 1:
                    first_entered.set()
                    await first_release.wait()
                    return first_value
                rebuild_entered.set()
                await rebuild_release.wait()
                return rebuilt_value
            finally:
                active -= 1

        async def filler_build():
            return "filler"

        async def b_build():
            b_entered.set()
            await b_release.wait()
            return "b"

        for key in fillers:
            await cache._cache_or_build(layer, key, filler_build)
        try:
            leader = asyncio.create_task(cache._cache_or_build(layer, "a", a_build))
            tasks.append(leader)
            await first_entered.wait()
            waiter = asyncio.create_task(_ready_call(waiter_ready, layer, "a", a_build))
            tasks.append(waiter)
            await waiter_ready.wait()
            b = asyncio.create_task(cache._cache_or_build(layer, "b", b_build))
            tasks.append(b)
            await b_entered.wait()
            gc.collect()
            # Queue A completion, filler hits, B completion, fresh A. The old
            # A waiter wakes only when the first A releases its context.
            first_release.set()
            tasks.extend(
                asyncio.create_task(cache._cache_or_build(layer, key, filler_build))
                for key in fillers
            )
            b_release.set()
            fresh = asyncio.create_task(cache._cache_or_build(layer, "a", a_build))
            tasks.append(fresh)
            await b
            await rebuild_entered.wait()
            assert "a" not in layer.store  # The replacement is still blocked.
            assert peak == 1, "same key has concurrent builders after eviction"
            rebuild_release.set()
            await asyncio.gather(*tasks)
            assert calls == 2
            assert waiter.result() is fresh.result() is rebuilt_value
            assert leader.result() is first_value
            assert len(layer.store) <= capacity
        finally:
            first_release.set()
            b_release.set()
            rebuild_release.set()
            await _finish_tasks(tasks)

    _run(scenario())


@pytest.mark.parametrize(
    "mode", ["success", "error", "cancel_leader", "cancel_waiter", "cancel_all_waiters"]
)
def test_same_key_builds_recover_from_failure_or_cancellation(mode: str):
    async def scenario():
        layer = cache.CacheLayer("test", OrderedDict(), max_size=2)
        entered, release = asyncio.Event(), asyncio.Event()
        ready = [asyncio.Event(), asyncio.Event()]
        value = object()
        calls = active = peak = 0
        tasks = []

        async def build():
            nonlocal calls, active, peak
            calls += 1
            attempt = calls
            active += 1
            peak = max(peak, active)
            try:
                if attempt == 1:
                    entered.set()
                    await release.wait()
                    if mode == "error":
                        raise ValueError("controlled builder failure")
                return value
            finally:
                active -= 1

        try:
            leader = asyncio.create_task(cache._cache_or_build(layer, "a", build))
            tasks.append(leader)
            await entered.wait()
            waiters = [
                asyncio.create_task(_ready_call(event, layer, "a", build))
                for event in ready
            ]
            tasks.extend(waiters)
            await asyncio.gather(*(event.wait() for event in ready))
            if mode == "cancel_leader":
                leader.cancel()
            elif mode == "cancel_waiter":
                waiters[0].cancel()
            elif mode == "cancel_all_waiters":
                for task in waiters:
                    task.cancel()
                await asyncio.gather(*waiters, return_exceptions=True)
                replacement_ready = asyncio.Event()
                tasks.append(
                    asyncio.create_task(
                        _ready_call(replacement_ready, layer, "a", build)
                    )
                )
                await replacement_ready.wait()
                gc.collect()
                assert calls == active == 1
            release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            expected_calls = 2 if mode in {"error", "cancel_leader"} else 1
            assert calls == expected_calls and peak == 1 and active == 0
            if mode == "error":
                assert isinstance(results[0], ValueError)
            elif mode == "cancel_leader":
                assert isinstance(results[0], asyncio.CancelledError)
            elif mode == "cancel_waiter":
                assert isinstance(results[1], asyncio.CancelledError)
            elif mode == "cancel_all_waiters":
                assert all(
                    isinstance(item, asyncio.CancelledError) for item in results[1:3]
                )
            successes = [
                item for item in results if not isinstance(item, BaseException)
            ]
            assert successes and all(item is value for item in successes)
            assert await cache._cache_or_build(layer, "a", build) is value
            assert calls == expected_calls
        finally:
            release.set()
            await _finish_tasks(tasks)

    _run(scenario())


def test_different_keys_build_concurrently():
    async def scenario():
        layer = cache.CacheLayer("test", OrderedDict(), max_size=2)
        entered = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()

        async def build(index):
            entered[index].set()
            await release.wait()
            return index

        tasks = [
            asyncio.create_task(
                cache._cache_or_build(layer, index, lambda index=index: build(index))
            )
            for index in range(2)
        ]
        try:
            await asyncio.gather(*(event.wait() for event in entered))
            release.set()
            assert await asyncio.gather(*tasks) == [0, 1]
        finally:
            release.set()
            await _finish_tasks(tasks)

    _run(scenario())


def test_cache_hits_update_lru_recency():
    async def scenario():
        layer = cache.CacheLayer("test", OrderedDict(), max_size=2)
        counts = {}

        async def get(key):
            async def build():
                counts[key] = counts.get(key, 0) + 1
                return object()

            value = await cache._cache_or_build(layer, key, build)
            assert len(layer.store) <= 2
            return value

        first = await get("a")
        await get("b")
        assert await get("a") is first
        await get("c")
        assert "a" in layer.store and "b" not in layer.store
        await get("b")
        assert counts == {"a": 1, "b": 2, "c": 1}

    _run(scenario())


@pytest.mark.parametrize("mode", ["success", "error", "cancel"])
def test_completed_requests_do_not_retain_unbounded_idle_keys(mode: str):
    class Key:
        pass

    async def scenario():
        capacity = 3
        layer = cache.CacheLayer("test", OrderedDict(), max_size=capacity)
        keys = weakref.WeakSet()

        async def request():
            key = Key()
            keys.add(key)

            async def build():
                if mode == "error":
                    raise ValueError("controlled builder failure")
                if mode == "cancel":
                    raise asyncio.CancelledError
                return "complete"

            with contextlib.suppress(ValueError, asyncio.CancelledError):
                await cache._cache_or_build(layer, key, build)

        for _round in range(2):
            for _ in range(100):
                await request()
            gc.collect()
            assert len(keys) <= (capacity if mode == "success" else 0)
            assert len(layer.store) <= capacity

    _run(scenario())
