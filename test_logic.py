#!/usr/bin/env python3
"""Self-checks for the non-trivial pure logic. Run: python3 test_logic.py"""
import numpy as np
import clip_backtrack as cb

def test_dedup():
    assert cb.dedup("tricked tricked tricked go") == "tricked go"           # word stutter
    assert cb.dedup("come down come down on this") == "come down on this"   # 2-word phrase
    assert cb.dedup("full force of full force of the win") == "full force of the win"  # 3-word
    # non-consecutive natural repeats left alone
    assert cb.dedup("my best friend and my best friend") == "my best friend and my best friend"
    assert cb.dedup("hello world") == "hello world"

def test_repetitive():
    assert cb.repetitive("tricked " * 10)                                   # hallucinated loop
    assert not cb.repetitive("I thought it was an incel angle because that is a real problem")
    assert not cb.repetitive("short phrase here")                           # too short to judge

def test_ringbuffer():
    orig = cb.SAMPLE_RATE
    try:
        cb.SAMPLE_RATE = 1
        b = cb.RingBuffer(5)                                    # capacity 5
        b.add(np.arange(3, dtype=np.float32))
        assert list(b.last(5)) == [0, 1, 2]
        b.add(np.arange(3, 7, dtype=np.float32))               # wraps
        assert list(b.last(5)) == [2, 3, 4, 5, 6]
        assert list(b.last(2)) == [5, 6]                       # last-N window
        b.add(np.arange(100, 120, dtype=np.float32))           # block > capacity
        assert list(b.last(5)) == [115, 116, 117, 118, 119]
        assert cb.RingBuffer(5).last(5).size == 0              # empty
    finally:
        cb.SAMPLE_RATE = orig

if __name__ == "__main__":
    test_dedup(); test_repetitive(); test_ringbuffer()
    print("all logic tests OK")
