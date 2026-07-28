#!/usr/bin/env python3
"""Self-check for AudioBuffer ring buffer. Run: python3 test_audiobuffer.py"""
import types, threading, numpy as np

# Load only the AudioBuffer class from the module, without importing its heavy
# deps (whisper, flask, sounddevice) or running module-level startup code.
import ast, textwrap
src = open("mac_clip_backtrack.py").read()
cls = next(n for n in ast.parse(src).body
           if isinstance(n, ast.ClassDef) and n.name == "AudioBuffer")
ns = {"np": np, "threading": threading, "RECORDING_PAUSED": False}
exec(compile(ast.Module([cls], []), "<AudioBuffer>", "exec"), ns)
AudioBuffer = ns["AudioBuffer"]

def test():
    b = AudioBuffer(sample_rate=1, max_buffer_seconds=5)  # capacity 5

    b.add_samples(np.arange(3, dtype=np.int16))           # [0 1 2]
    assert list(b.get_audio_data(5)) == [0, 1, 2]

    b.add_samples(np.arange(3, 7, dtype=np.int16))        # [3 4 5 6] -> wraps
    assert list(b.get_audio_data(5)) == [2, 3, 4, 5, 6], list(b.get_audio_data(5))

    assert list(b.get_audio_data(2)) == [5, 6]            # last-N window

    b.add_samples(np.arange(100, 120, dtype=np.int16))    # block > capacity
    assert list(b.get_audio_data(5)) == [115, 116, 117, 118, 119]

    assert AudioBuffer(1, 5).get_audio_data(5).size == 0  # empty
    print("AudioBuffer OK")

if __name__ == "__main__":
    test()
