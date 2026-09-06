import os
import re
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.drain()
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


# --- host prompt cache (--cache-ram): switching to a better cached prompt ---

PREFIX = (
    "You are a helpful assistant. Answer briefly and politely, and always stay on topic. "
    "The following conversation takes place in a small village near the mountains. "
)

STORY = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom. "
)

OTHER = (
    "Quarterly report: revenue grew by twelve percent while costs stayed flat. "
    "The board approved the new budget and thanked the finance team for their work. "
)


def n_tokens(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return len(res.body["tokens"])


def completion(prompt: str):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
    })
    assert res.status_code == 200, res.body
    return res.body["timings"]


def saved_sizes_mib(log_text: str):
    # "saving prompt with length 123, total state size = 1.234 MiB (draft: 0.000 MiB)"
    return [float(m) for m in re.findall(r"saving prompt with length \d+, total state size = ([0-9.]+) MiB", log_text)]


# Returning to a long conversation from a shorter one that shares the same prefix (e.g. the system
# prompt): the slot keeps most of its own short context (f_keep >= 0.5), but the cache holds the full
# long prompt, which covers far more of the new request - it must be restored instead of
# re-processing everything after the shared prefix.
def test_restore_longer_prompt_from_cache():
    global server
    server.n_slots = 1
    server.n_ctx = 2048
    server.cache_ram = 100
    server.start()
    log = LogReader(server.log_path)

    long_prompt = PREFIX + STORY * 3
    short_prompt = PREFIX + "The cat sat on the mat."

    n_long = n_tokens(long_prompt)
    n_prefix = n_tokens(PREFIX)
    n_short = n_tokens(short_prompt)

    # the return to the long prompt must take the "slot is similar enough, keep it" path:
    # f_sim = n_prefix / n_long > 0.1 (slot selected by LCP similarity) and
    # f_keep = n_prefix / (n_short + n_predict) >= 0.5 (slot keeps most of its context)
    assert n_prefix / n_long > 0.1
    assert n_prefix / (n_short + server.n_predict) >= 0.5

    t = completion(long_prompt)
    assert t["cache_n"] == 0
    assert t["prompt_n"] in (n_long, n_long + 1)  # BOS

    # the short prompt keeps less than half of the slot -> the long prompt is saved to the cache
    t = completion(short_prompt)
    assert t["cache_n"] > 0
    assert len(saved_sizes_mib(log.drain())) == 1

    # back to the long prompt (plus a new turn): restored from the cache, only the new tokens are processed
    t = completion(long_prompt + " The knight finally reached the castle gates.")
    assert t["cache_n"] >= n_long - 2, f"cache_n = {t['cache_n']}, expected about {n_long}"


# Saving the current slot before restoring a cached state must not evict the very state that is
# about to be restored: with a cache that holds either state but not both, alternating between two
# prompts must still restore the one that was cached.
def test_restore_is_not_evicted_by_save():
    global server
    server.n_slots = 1
    server.n_ctx = 2048
    cache_mib = 1  # fits either prompt state below (~0.77 MiB and ~0.43 MiB at ~650 B/token), not both
    server.cache_ram = cache_mib
    server.start()
    log = LogReader(server.log_path)

    long_prompt = STORY * 7    # ~1232 tokens
    other_prompt = OTHER * 8   # ~696 tokens

    n_long = n_tokens(long_prompt)

    t = completion(long_prompt)
    assert t["cache_n"] == 0

    # unrelated prompt (no common prefix beyond BOS): the long prompt is saved to the cache
    t = completion(other_prompt)
    assert t["cache_n"] <= 1
    sizes = saved_sizes_mib(log.drain())
    assert len(sizes) == 1
    size_long = sizes[0]

    # back to the long prompt: the other prompt is saved (it does not fit next to the long one),
    # the long prompt is restored
    t = completion(long_prompt + " The knight finally reached the castle gates.")
    text = log.drain()
    sizes = saved_sizes_mib(text)
    assert len(sizes) == 1
    size_other = sizes[0]

    # calibration of the scenario itself: each state fits alone, both do not fit together
    assert size_long < cache_mib and size_other < cache_mib, f"state sizes {size_long} / {size_other} MiB do not fit a {cache_mib} MiB cache"
    assert size_long + size_other > cache_mib, f"state sizes {size_long} + {size_other} MiB fit together - the scenario needs a smaller cache"

    assert t["cache_n"] >= n_long - 2, f"cache_n = {t['cache_n']}, expected about {n_long}"
