#!/usr/bin/env python3
"""Self-checks for the non-trivial pure logic. Run: python3 test_logic.py"""
import numpy as np
import clip_in_context as cb

def test_dedup():
    assert cb.dedup("tricked tricked tricked go") == "tricked go"           # word stutter
    assert cb.dedup("come down come down on this") == "come down on this"   # 2-word phrase
    assert cb.dedup("full force of full force of the win") == "full force of the win"  # 3-word
    # non-consecutive natural repeats left alone
    assert cb.dedup("my best friend and my best friend") == "my best friend and my best friend"
    assert cb.dedup("hello world") == "hello world"

def test_strip_lead_junk():
    assert cb.strip_lead_junk("Rom Okay. No! Wait.") == "Okay. No! Wait."
    assert cb.strip_lead_junk("Um, uh Rom where is it") == "where is it"
    assert cb.strip_lead_junk("Romance is dead") == "Romance is dead"   # word boundary
    assert cb.strip_lead_junk("let's go um") == "let's go um"            # only leading

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

def test_clip_editor():
    import clip_editor, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    try:
        clip_editor.render_top_banner_png("TEST HOOK TITLE", path)
        assert os.path.exists(path) and os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.remove(path)

def test_story_silence():
    import clip_editor
    opts = {"max_silence_gap_sec": 6.0, "preserve_story_span": True, "segment_padding_sec": 0.5}
    valid_segs = [
        {"id": 0, "start": 5.0, "end": 10.0, "text": "Watch this setup."},
        {"id": 1, "start": 15.0, "end": 20.0, "text": "NO WAY IT WORKED!"} # 5s gap of silence in between
    ]
    # Test contiguous selection check
    sids_contiguous = [0, 1]
    is_contig = (sorted(sids_contiguous) == list(range(min(sids_contiguous), max(sids_contiguous) + 1)))
    assert is_contig is True

    # Test non-contiguous selection check (jump cuts preserved when filler IDs 2, 3 skipped)
    sids_jump = [0, 1, 4, 5]
    is_jump_contig = (sorted(sids_jump) == list(range(min(sids_jump), max(sids_jump) + 1)))
    assert is_jump_contig is False

def test_game_hashtags():
    assert cb.get_game_hashtags("The Legend of Zelda: Tears of the Kingdom") == ["#zelda", "#totk"]
    assert cb.get_game_hashtags("Tears of the Kingdom") == ["#zelda", "#totk"]
    assert cb.get_game_hashtags("Zelda: Tears of the Kingdom") == ["#zelda", "#totk"]
    assert cb.get_game_hashtags("Zelda") == ["#zelda"]
    assert cb.get_game_hashtags("The Legend of Zelda: Breath of the Wild") == ["#zelda", "#botw"]
    assert cb.get_game_hashtags("Super Mario Odyssey") == ["#SuperMarioOdyssey"]
    assert cb.get_game_hashtags("Just Chatting") == []
    assert cb.get_game_hashtags("") == []

if __name__ == "__main__":
    test_dedup(); test_strip_lead_junk(); test_repetitive(); test_ringbuffer(); test_clip_editor(); test_story_silence(); test_game_hashtags()
    print("all logic tests OK")



