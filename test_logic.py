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

def test_find_latest_skips_editor_output():
    import tempfile, os
    d = tempfile.mkdtemp()
    orig = cb.cfg["obs_clips_dir"]
    try:
        cb.cfg["obs_clips_dir"] = d
        for i, n in enumerate(["Clip.mp4", "Clip_STORY.mp4", "Clip_edited.mp4", "Clip_story_raw.mp4"]):
            p = os.path.join(d, n); open(p, "w").close(); os.utime(p, (1000 + i, 1000 + i))
        assert os.path.basename(cb.find_latest_clip()) == "Clip.mp4"   # newer editor files ignored
    finally:
        cb.cfg["obs_clips_dir"] = orig

def test_draw_image_path_sandbox():
    import draw_showcase as ds, tempfile, os
    d = tempfile.mkdtemp()
    orig = ds.SCREENSHOTS_DIR
    try:
        ds.SCREENSHOTS_DIR = d
        ok = os.path.join(d, "a.png"); open(ok, "w").close()
        assert ds._screenshot_path(ok) == os.path.realpath(ok)
        assert ds._screenshot_path(os.path.join(d, "..", "x.png")) is None     # traversal
        assert ds._screenshot_path(os.path.abspath(__file__)) is None           # not a png / outside
        assert ds._screenshot_path("") is None
    finally:
        ds.SCREENSHOTS_DIR = orig

def test_game_hashtags():
    assert cb.get_game_hashtags("The Legend of Zelda: Tears of the Kingdom") == ["#zelda", "#totk"]
    assert cb.get_game_hashtags("Tears of the Kingdom") == ["#zelda", "#totk"]
    assert cb.get_game_hashtags("Zelda: Tears of the Kingdom") == ["#zelda", "#totk"]
    assert cb.get_game_hashtags("Zelda") == ["#zelda"]
    assert cb.get_game_hashtags("The Legend of Zelda: Breath of the Wild") == ["#zelda", "#botw"]
    assert cb.get_game_hashtags("Super Mario Odyssey") == ["#SuperMarioOdyssey"]
    assert cb.get_game_hashtags("Just Chatting") == []
    assert cb.get_game_hashtags("") == []

def test_build_metadata():
    orig = cb.cfg.get("streamer_name")
    try:
        cb.cfg["streamer_name"] = "Mesut"
        title, tags, desc = cb.build_metadata("Missed by Inches", "oh shit no", "Tears of the Kingdom")
        assert title == "Missed by Inches #Shorts #zelda #totk #Gaming #TwitchClips #ShortsViral"
        assert tags[0] == "Tears of the Kingdom" and "totk" in tags and tags[-1] == "Mesut"
        assert '"oh s*** no"' in desc                                  # transcript is censored
        long_title, _, _ = cb.build_metadata("x" * 90, "", "")
        assert len(long_title) <= 100 and long_title.endswith("#Shorts")   # whole hashtags only
    finally:
        cb.cfg["streamer_name"] = orig

def test_next_publish_slot():
    from datetime import datetime
    now = datetime(2026, 9, 26, 14, 50).astimezone()
    s = cb.next_publish_slot(["12:00", "15:00", "18:00"], set(), now)
    assert (s.day, s.hour) == (26, 18)                        # 15:00 is <30 min away
    taken = {s.isoformat()}
    s2 = cb.next_publish_slot(["12:00", "15:00", "18:00"], taken, now)
    assert (s2.day, s2.hour) == (27, 12)                      # next free slot rolls to tomorrow

def test_parse_titles():
    assert cb.parse_titles('1. Missed by Inches\n2) "Clutch Save"\n- Lost Treasure Hunt\nTitles:') == \
        ["Missed by Inches", "Clutch Save", "Lost Treasure Hunt"]

if __name__ == "__main__":
    test_dedup(); test_strip_lead_junk(); test_repetitive(); test_ringbuffer(); test_clip_editor(); test_find_latest_skips_editor_output(); test_draw_image_path_sandbox(); test_game_hashtags(); test_build_metadata(); test_next_publish_slot(); test_parse_titles()
    print("all logic tests OK")



