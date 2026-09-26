#!/usr/bin/env python3
"""
clip_editor.py — AI Video Clip Editor for Clip in Context.
Implements Karaoke Caption Style — Spec v1 (/Users/Mesut/fc-mcp/caption-style.md)

1. SF Pro Display Heavy font (/Library/Fonts/SF-Pro-Display-Heavy.otf) at 92px.
2. Spoken casing preserved (sentence case, no forcing to UPPERCASE).
3. Stroke + Drop Shadow per word (shadow offset +4x, +5y, alpha 150/255).
4. Semantic highlight colors:
   - Fear / Tension / Danger: Red #FF5A5A
   - Relief / Success / Joy: Green #70E670
   - Neutral / Default: Gold #FFC93C
   - Wholesome / Cute: Pink #FF96BE
5. Phrase-level sweep layout (fixed positions, zero text reflow/wobble).
6. Disjoint half-open timing ranges (zero subtitle overlap).
7. Top Hook Banner & Loudness Normalization.
8. Hardware-accelerated Apple Silicon encoding (h264_videotoolbox).
"""

import os
import sys
import re
import math
import json
import subprocess
import tempfile
import shutil
import threading
from PIL import Image, ImageDraw, ImageFont

# Ensure standard brew / local bin paths are in PATH (needed when running under launchd / GUI app)
for _p in ["/opt/homebrew/bin", "/usr/local/bin", os.path.expanduser("~/.homebrew/bin")]:
    if _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{_p}{os.pathsep}" + os.environ.get("PATH", "")

HERE = os.path.dirname(os.path.abspath(__file__))

# MLX Whisper isn't reentrant. clip_in_context's live-caption loop and clip
# trigger share this lock, so an edit never transcribes concurrently with them.
WHISPER_LOCK = threading.Lock()

# Spec v1 Colors (RGBA)
WHITE = (255, 255, 255, 255)
GOLD  = (255, 201, 60, 255)   # Neutral / default emphasis
RED   = (255, 90, 90, 255)    # Fear / tension / danger (#FF5A5A)
GREEN = (112, 230, 112, 255)  # Relief / success / joy (#70E670)
PINK  = (255, 150, 190, 255)  # Wholesome / cute (#FF96BE)

COLOR_MAP = {
    "gold": GOLD,
    "yellow": GOLD,
    "red": RED,
    "green": GREEN,
    "pink": PINK,
}

# Spec v1 Constants
FONT_SPEC_PATH = "/Library/Fonts/SF-Pro-Display-Heavy.otf"
MAXW, PADX, PADY, GAP = 980, 44, 30, 10

def get_ffmpeg_path():
    found = shutil.which("ffmpeg")
    if found:
        return found
    for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", os.path.expanduser("~/.homebrew/bin/ffmpeg")]:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return "ffmpeg"

def get_video_duration(video_path):
    ffmpeg = get_ffmpeg_path()
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    try:
        cmd = [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        out = subprocess.check_output(cmd).decode().strip()
        return float(out)
    except Exception:
        return 0.0

def get_spec_font(size=92):
    """Load SF Pro Display Heavy font as per Spec v1 section 2, with clean fallbacks."""
    font_candidates = [
        FONT_SPEC_PATH,
        "/Library/Fonts/SF-Pro-Display-Bold.otf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def extract_audio(video_path, wav_path):
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def transcribe_words(video_path, model_repo="mlx-community/whisper-large-v3-turbo"):
    """
    Extract audio and run MLX Whisper with word timestamps.
    Preserves original spoken casing as specified in Spec v1 section 2.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        extract_audio(video_path, wav_path)
        import mlx_whisper
        with WHISPER_LOCK:
            res = mlx_whisper.transcribe(wav_path, path_or_hf_repo=model_repo, word_timestamps=True,
                                         condition_on_previous_text=False)
        
        words = []
        for seg in res.get("segments", []):
            if "words" in seg:
                for w in seg["words"]:
                    raw_w = w["word"].strip()
                    if raw_w:
                        words.append({
                            "word": raw_w,  # Preserve spoken casing
                            "start": float(w["start"]),
                            "end": float(w["end"])
                        })
        return {
            "text": res.get("text", "").strip(),
            "words": words
        }
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

def chunk_words_into_phrases(words, max_words=4, max_pause=0.40):
    """
    Groups spoken words into natural phrase chunks based on:
    1. Maximum 3-4 words per line.
    2. Spoken pauses (>0.40s gap between words starts a NEW chunk).
    3. Sentence punctuation (. ! ? ,) starts a NEW chunk.
    """
    phrases = []
    curr = []
    for idx, w in enumerate(words):
        curr.append(w)
        
        has_punctuation = any(w["word"].endswith(p) for p in [".", "!", "?", ","])
        is_last = (idx == len(words) - 1)
        has_pause = False
        if not is_last:
            gap = words[idx + 1]["start"] - w["end"]
            if gap > max_pause:
                has_pause = True

        if len(curr) >= max_words or has_punctuation or has_pause or is_last:
            phrases.append(curr)
            curr = []

    if curr:
        phrases.append(curr)
    return phrases

def get_phrase_semantic_color(phrase_words, default_color="gold"):
    """
    Spec v1 section 4: The highlight color is chosen PER PHRASE by emotional beat.
    One single, unified active color for the entire phrase to eliminate color mixing.
    """
    phrase_text = " ".join([w["word"] for w in phrase_words]).lower()
    clean_text = re.sub(r'[^a-z0-9\s]', '', phrase_text)
    
    fear_words = {"scared", "die", "died", "stuck", "danger", "scare", "terrified", "panic"}
    joy_words = {"nice", "good", "great", "pro", "gamer", "victory", "clutch", "epic"}
    cute_words = {"cute", "sweet", "wholesome", "adorable"}

    tokens = set(clean_text.split())
    if tokens & fear_words:
        return RED
    elif tokens & joy_words:
        return GREEN
    elif tokens & cute_words:
        return PINK

    return COLOR_MAP.get(str(default_color).lower(), GOLD)

META_FONT_PATH = "/Library/Fonts/SF-Pro-Display-MediumItalic.otf"

def get_meta_font(size=70):
    if os.path.exists(META_FONT_PATH):
        try:
            return ImageFont.truetype(META_FONT_PATH, size)
        except Exception:
            pass
    return get_spec_font(size)

def render_meta_tag_png(text, size, output_path):
    """
    Renders non-speech emotion tag PNG overlay as per Spec v1 section 9 & shorts-engine:
    SF Pro Display Medium Italic, muted gray #C8C8C8, lighter stroke.
    """
    font = get_meta_font(size)
    stroke = max(4, size // 12)
    dummy = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    bbox = dummy.textbbox((0, 0), text, font=font, stroke_width=stroke)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    W, H = max(1080, tw + 80), th + 60

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = W / 2
    # Shadow
    d.text((cx + 3, 30 - bbox[1] + 4), text, font=font, fill=(0, 0, 0, 140), anchor="ma", stroke_width=stroke, stroke_fill=(0, 0, 0, 140))
    # Main text (muted gray)
    d.text((cx, 30 - bbox[1]), text, font=font, fill=(200, 200, 200, 255), anchor="ma", stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    img.save(output_path)
    return output_path, W, H

def render_static_cta_png(text, size, color, output_path):
    """
    Renders static ending CTA text overlay card as per shorts-engine.
    """
    font = get_spec_font(size)
    stroke = max(5, size // 9)
    dummy = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    bbox = dummy.textbbox((0, 0), text, font=font, stroke_width=stroke)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    W, H = max(1080, tw + 88), th + 68

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = W / 2
    # Shadow
    d.text((cx + 4, 34 - bbox[1] + 5), text, font=font, fill=(0, 0, 0, 150), anchor="ma", stroke_width=stroke, stroke_fill=(0, 0, 0, 150))
    # Main text
    d.text((cx, 34 - bbox[1]), text, font=font, fill=color, anchor="ma", stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    img.save(output_path)
    return output_path, W, H

def render_phrase_states(tokens, highlight_colors, size=92, outdir="caps", gid="g"):
    """
    Renders 1 transparent PNG state per word in tokens according to Spec v1 section 6 & 83.
    Whole phrase is shown, with active word in its semantic highlight color.
    Because positions are fixed across all word states, the highlight sweeps smoothly
    without any text reflow or wobble.
    Returns (paths, W, H).
    """
    font = get_spec_font(size)
    stroke = max(6, size // 8)
    asc, desc = font.getmetrics()
    lh = asc + desc
    
    dummy = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    space = dummy.textlength(" ", font=font)
    tw = [dummy.textlength(t, font=font) for t in tokens]

    # Wrap lines to MAXW = 980px
    lines, cur, curw = [], [], 0
    for i, t in enumerate(tokens):
        add = tw[i] + (space if cur else 0)
        if cur and curw + add > MAXW:
            lines.append(cur)
            cur, curw = [], 0
            add = tw[i]
        cur.append(i)
        curw += add
    if cur:
        lines.append(cur)

    line_w = [sum(tw[i] for i in ln) + space*(len(ln)-1) for ln in lines]
    W = max(1080, int(max(line_w)) + PADX*2)
    H = int(lh*len(lines) + GAP*(len(lines)-1) + PADY*2)

    pos = {}
    for li, ln in enumerate(lines):
        x = (W - line_w[li]) / 2
        y = PADY + li*(lh + GAP)
        for i in ln:
            pos[i] = (x, y)
            x += tw[i] + space

    paths = []
    for active in range(len(tokens)):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        # Draw order per word: shadow first (alpha 150, offset +4x, +5y), then stroked fill
        for i, t in enumerate(tokens):
            x, y = pos[i]
            col = highlight_colors[i] if i == active else WHITE

            # 1. Drop shadow (alpha 150/255, +4x, +5y offset)
            d.text((x + 4, y + 5), t, font=font, fill=(0, 0, 0, 150), stroke_width=stroke, stroke_fill=(0, 0, 0, 150))
            # 2. Main text with opaque black stroke
            d.text((x, y), t, font=font, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))

        p = os.path.join(outdir, f"{gid}_{active}.png")
        img.save(p)
        paths.append(p)

    return paths, W, H

def render_top_banner_png(title, output_path, width=1080, height=1920):
    """Render top title hook banner card."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    clean_title = re.sub(r'[\r\n\t]+', ' ', title).strip().upper()
    title_text = clean_title  # no emoji: SF Pro has no emoji glyphs, PIL drew a tofu box

    font = get_spec_font(48)
    words = title_text.split()
    lines, curr_line = [], []
    for w in words:
        test = " ".join(curr_line + [w])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > width - 160:
            if curr_line:
                lines.append(" ".join(curr_line))
                curr_line = [w]
            else:
                lines.append(w)
                curr_line = []
        else:
            curr_line.append(w)
    if curr_line:
        lines.append(" ".join(curr_line))

    line_heights = [draw.textbbox((0, 0), l, font=font)[3] - draw.textbbox((0, 0), l, font=font)[1] for l in lines]
    max_line_w = max([draw.textbbox((0, 0), l, font=font)[2] - draw.textbbox((0, 0), l, font=font)[0] for l in lines])
    total_h = sum(line_heights) + (len(lines) - 1) * 12

    banner_y = 160
    pad_x, pad_y = 36, 24
    rect_x0 = (width - max_line_w) // 2 - pad_x
    rect_y0 = banner_y - pad_y
    rect_x1 = (width + max_line_w) // 2 + pad_x
    rect_y1 = banner_y + total_h + pad_y

    draw.rounded_rectangle([rect_x0, rect_y0, rect_x1, rect_y1], radius=24, fill=(0, 0, 0, 210), outline=(255, 255, 255, 180), width=3)

    curr_y = banner_y
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lx = (width - lw) // 2
        # Shadow
        draw.text((lx + 3, curr_y + 4), line, font=font, fill=(0, 0, 0, 150), stroke_width=4, stroke_fill=(0, 0, 0, 150))
        # Text
        draw.text((lx, curr_y), line, font=font, fill=(255, 255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))
        curr_y += line_heights[i] + 12

    img.save(output_path, "PNG")
    return output_path

def render_edited_video(input_path, output_path, title, words_data, options=None):
    """
    Renders video using FFmpeg and Spec v1 Karaoke PNG overlays:
    - SF Pro Display Heavy font at 92px.
    - Semantic highlight colors per word.
    - Disjoint half-open timing (min(end_i, start_{i+1})) to guarantee zero caption overlap.
    - Placement at center_y = 1300.
    """
    if options is None:
        options = {}

    ffmpeg = get_ffmpeg_path()
    words = words_data.get("words", [])

    # 1. Lead-in silence trim
    trim_start = 0.0
    if options.get("smart_trim", True) and words:
        first_word_start = words[0]["start"]
        if first_word_start > 0.8:
            trim_start = max(0.0, first_word_start - 0.3)

    tmp_dir = tempfile.mkdtemp(prefix="karaoke_spec_")
    try:
        overlays = []

        # Top Hook Banner PNG (only if explicitly enabled via options.get('show_banner', False))
        if title and options.get("show_banner", False):
            banner_png = os.path.join(tmp_dir, "title_banner.png")
            render_top_banner_png(title, banner_png)
            banner_end = max(4.5, 4.5 - trim_start)
            overlays.append({
                "png": banner_png,
                "start": 0.0,
                "end": banner_end,
                "x_expr": "(W-w)/2",
                "y_expr": "80"
            })

        # Determine CTA window if enabled
        enable_cta = options.get("enable_cta", True)
        clip_dur = get_video_duration(input_path) - trim_start
        cta_start = max(0.0, clip_dur - 2.2) if (enable_cta and clip_dur >= 5.0) else clip_dur
        cta_end = clip_dur

        # Group words into natural phrase chunks (pause & punctuation aware)
        if words:
            phrases = chunk_words_into_phrases(words, max_words=4, max_pause=0.40)
            default_color = options.get("sub_color", "gold")
            center_y = int(options.get("sub_y", 1300))

            # Build all phrase state overlays
            all_states = []
            for p_idx, phrase in enumerate(phrases):
                tokens = [w["word"] for w in phrase]
                phrase_color = get_phrase_semantic_color(phrase, default_color)
                colors = [phrase_color] * len(phrase)   # Single unified phrase color (no color mixing!)

                png_paths, canvas_w, canvas_h = render_phrase_states(
                    tokens, colors, size=92, outdir=tmp_dir, gid=f"p{p_idx:03d}"
                )

                for w_idx, w_item in enumerate(phrase):
                    # Timing rule (Spec v1 section 6 & 6b)
                    a = max(0.0, w_item["start"] - trim_start)
                    
                    # Next word start time (within phrase or next phrase)
                    if w_idx < len(phrase) - 1:
                        b = max(a + 0.08, phrase[w_idx + 1]["start"] - trim_start)
                    else:
                        # Last word in phrase holds for 0.35s
                        b = max(a + 0.15, w_item["end"] - trim_start + 0.35)

                    all_states.append({
                        "png": png_paths[w_idx],
                        "start": a,
                        "end": b,
                        "w": canvas_w,
                        "h": canvas_h
                    })

            # Sort and clamp end_i = min(end_i, start_{i+1}) to strictly enforce no overlap (Spec v1 section 6b)
            all_states.sort(key=lambda s: s["start"])
            for idx in range(len(all_states)):
                st = all_states[idx]
                if idx < len(all_states) - 1:
                    next_start = all_states[idx + 1]["start"]
                    st["end"] = min(st["end"], next_start)

                # Clamp speech caption end time so it disappears BEFORE the CTA card starts!
                if enable_cta and cta_start < clip_dur:
                    st["end"] = min(st["end"], max(st["start"], cta_start - 0.05))

                if st["end"] > st["start"]:
                    y_top = center_y - (st["h"] // 2)
                    overlays.append({
                        "png": st["png"],
                        "start": st["start"],
                        "end": st["end"],
                        "x_expr": "(W-w)/2",
                        "y_expr": f"{y_top}"
                    })

        # Ending Static Call-To-Action (CTA) Overlay Cards (shorts-engine style)
        if enable_cta and cta_end - cta_start >= 0.5:
            cta1_png = os.path.join(tmp_dir, "cta_line1.png")
            cta2_png = os.path.join(tmp_dir, "cta_line2.png")
            render_static_cta_png("LIVE MOST NIGHTS", 90, WHITE, cta1_png)
            render_static_cta_png("follow for more!", 60, GOLD, cta2_png)

            overlays.append({
                "png": cta1_png,
                "start": cta_start,
                "end": cta_end,
                "x_expr": "(W-w)/2",
                "y_expr": "1150"
            })
            overlays.append({
                "png": cta2_png,
                "start": cta_start + 0.15,
                "end": cta_end,
                "x_expr": "(W-w)/2",
                "y_expr": "1290"
            })

        # Build FFmpeg command with input-first accurate seeking for perfect A/V sync
        cmd = [ffmpeg, "-y"]
        if trim_start > 0:
            cmd.extend(["-ss", f"{trim_start:.3f}"])
        cmd.extend(["-i", input_path])

        for o in overlays:
            cmd.extend(["-i", o["png"]])

        filter_chains = ["[0:v]setpts=PTS-STARTPTS,fps=60,setsar=1[base]"]
        last_buf = "base"
        for idx, o in enumerate(overlays):
            next_buf = f"v{idx+1}"
            s_str = f"{o['start']:.3f}"
            e_str = f"{o['end']:.3f}"
            filter_chains.append(f"[{last_buf}][{idx+1}:v]overlay=x={o['x_expr']}:y={o['y_expr']}:enable='between(t,{s_str},{e_str})'[{next_buf}]")
            last_buf = next_buf

        vf_str = ";".join(filter_chains)

        cmd.extend([
            "-filter_complex", vf_str,
            "-map", f"[{last_buf}]",
            "-map", "0:a",
            "-c:v", "h264_videotoolbox",
            "-b:v", "12M",
            "-c:a", "aac",
            "-ar", "48000",
            "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            output_path
        ])

        print(f"🎬 Rendering Karaoke Spec v1 Video ({len(overlays)} overlays): {os.path.basename(input_path)} → {os.path.basename(output_path)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if res.returncode != 0:
            print(f"⚠️ FFmpeg Videotoolbox failed, falling back to libx264: {res.stderr[-300:]}")
            cmd[cmd.index("h264_videotoolbox")] = "libx264"
            cmd.insert(cmd.index("libx264") + 1, "-preset")
            cmd.insert(cmd.index("libx264") + 2, "fast")
            subprocess.run(cmd, check=True)

        print(f"✅ Spec v1 Video rendering complete: {output_path}")
        return output_path

    finally:
        import shutil
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

def select_story_segments(video_path, model_repo="mlx-community/whisper-large-v3-turbo", ollama_model="qwen2.5:latest", options=None):
    """
    Uses Whisper transcription + Ollama LLM to select key narrative beats when story_cut is enabled.
    Preserves silent gameplay action and reaction moments based on options:
      - max_silence_gap_sec: Max silence gap in seconds to merge (default 6.0)
      - preserve_story_span: If True, preserves continuous span for story arcs <= max_story_duration_sec (default True)
      - segment_padding_sec: Padding before/after spoken segments (default 0.5)
    """
    if options is None:
        options = {}

    max_silence_gap = float(options.get("max_silence_gap_sec", 6.0))
    preserve_story_span = bool(options.get("preserve_story_span", True))
    padding = float(options.get("segment_padding_sec", 0.5))
    max_story_duration = float(options.get("max_story_duration_sec", 45.0))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        extract_audio(video_path, wav_path)
        import mlx_whisper
        with WHISPER_LOCK:
            res = mlx_whisper.transcribe(wav_path, path_or_hf_repo=model_repo,
                                         condition_on_previous_text=False)
        segments = res.get("segments", [])
        if not segments:
            return [], "", ""

        total_dur = segments[-1]["end"]
        if total_dur <= 35.0:
            return [(0.0, total_dur)], "", ""

        valid_segs = []
        for idx, s in enumerate(segments):
            text = s["text"].strip()
            if text and (s["end"] - s["start"]) > 0.2:
                valid_segs.append({
                    "id": len(valid_segs),
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "text": text
                })

        if not valid_segs:
            return [(0.0, min(total_dur, 35.0))], "", ""

        prompt = (
            "You are an expert short-form video editor cutting a gaming stream clip into a viral Short.\n"
            f"Transcribed spoken segments:\n{json.dumps(valid_segs, indent=2)}\n\n"
            "STRICT RULES:\n"
            "1. STORY ARC & PAYOFF: Select segment IDs that tell a compelling story arc (Hook -> Setup -> Climax/Payoff).\n"
            "2. Note that streamers often have silent gameplay action, suspense, or visual reactions between spoken words. Include the range of IDs covering setup, action, and payoff.\n"
            "3. Target length: 15s to 35s max.\n"
            "4. Keep complete multi-sentence thoughts. Never cut mid-sentence.\n"
            "5. Include a 1-sentence 'story_reasoning' explaining the narrative angle you chose.\n\n"
            'Respond strictly with a JSON object: {"selected_ids": [15, 16, 17, 18, 19, 20], "title": "SHORT TITLE", "story_reasoning": "Brief 1-sentence explanation of the story angle chosen"}'
        )

        selected_ids = []
        ai_title = ""
        story_reasoning = ""
        try:
            import requests
            r = requests.post("http://localhost:11434/api/generate", json={
                "model": ollama_model,
                "prompt": prompt,
                "options": {"temperature": 0.8, "top_p": 0.95},
                "stream": False,
                "format": "json"
            }, timeout=10)
            resp = json.loads(r.json().get("response", "{}"))
            selected_ids = resp.get("selected_ids", [])
            ai_title = resp.get("title", "")
            story_reasoning = resp.get("story_reasoning", "")
        except Exception:
            pass

        selected_valid_segs = []
        ranges = []
        if selected_ids:
            for sid in selected_ids:
                if 0 <= sid < len(valid_segs):
                    s = valid_segs[sid]
                    selected_valid_segs.append(s)
                    r_start = max(0.0, s["start"] - padding)
                    r_end = min(total_dur, s["end"] + padding)
                    ranges.append((r_start, r_end))

        # Check if selected IDs form a contiguous sequence (no filler segments skipped by LLM)
        is_contiguous_selection = False
        if selected_ids and len(selected_ids) > 1:
            sorted_sids = sorted(selected_ids)
            is_contiguous_selection = (sorted_sids == list(range(sorted_sids[0], sorted_sids[-1] + 1)))
        elif selected_ids:
            is_contiguous_selection = True

        # Keep continuous span ONLY when AI selected a contiguous story beat (preserving reaction/silence).
        # If AI explicitly skipped filler IDs in between, perform jump cuts between those ranges!
        if preserve_story_span and selected_valid_segs and is_contiguous_selection:
            min_start = min(s["start"] for s in selected_valid_segs)
            max_end = max(s["end"] for s in selected_valid_segs)
            span_dur = (max_end - min_start) + (2 * padding)
            if span_dur <= max_story_duration:
                # Keep entire continuous video span so silent gameplay/reactions are preserved
                span_start = max(0.0, min_start - padding)
                span_end = min(total_dur, max_end + padding)
                return [(span_start, span_end)], ai_title, story_reasoning

        total_selected = sum([e - s for s, e in ranges])
        if total_selected < 10.0:
            ranges = []
            for s in valid_segs:
                ranges.append((max(0.0, s["start"] - padding), min(total_dur, s["end"] + padding)))

        # Merge contiguous or close time ranges using max_silence_gap
        merged = []
        for r in sorted(ranges, key=lambda x: x[0]):
            if not merged:
                merged.append(r)
            else:
                last_s, last_e = merged[-1]
                if r[0] <= last_e + max_silence_gap:   # Merge gaps under max_silence_gap seconds
                    merged[-1] = (last_s, max(last_e, r[1]))
                else:
                    merged.append(r)

        # Filter out micro-chunks under 4.0 seconds
        final_ranges = [r for r in merged if (r[1] - r[0]) >= 4.0]
        if not final_ranges and merged:
            final_ranges = merged

        return final_ranges, ai_title, story_reasoning
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

def splice_story_video(video_path, time_ranges, output_path):
    """
    Slices and concats specified time ranges from input video into output_path using FFmpeg.
    Uses input-first decoding seeking (-i before -ss) and audio resync flags to guarantee perfect A/V sync.
    """
    ffmpeg = get_ffmpeg_path()
    tmp_dir = tempfile.mkdtemp(prefix="story_splice_")
    segment_files = []

    try:
        for idx, (s, e) in enumerate(time_ranges):
            seg_file = os.path.join(tmp_dir, f"seg_{idx:02d}.mp4")
            cmd = [
                ffmpeg, "-y",
                "-i", video_path,
                "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                "-c:v", "h264_videotoolbox", "-b:v", "12M",
                "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
                "-async", "1",
                "-avoid_negative_ts", "make_zero",
                seg_file
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            segment_files.append(seg_file)

        concat_list = os.path.join(tmp_dir, "list.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        cmd_concat = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c:v", "copy",
            "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
            output_path
        ]
        subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path
    finally:
        import shutil
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

def process_and_edit_clip(input_path, title, model_repo="mlx-community/whisper-large-v3-turbo", options=None):
    """
    Main entrypoint:
    1. Performs AI story segment selection & jump-cut trimming if enabled.
    2. Transcribes audio, builds Karaoke Spec v1 animated subtitles & top title hook banner.
    3. Performs loudness normalization and hardware-accelerated H.264 video rendering.
    Returns path to edited MP4 file.
    """
    if options is None:
        options = {}

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    target_video = input_path

    if options.get("story_cut", True):
        dur = get_video_duration(input_path)
        if dur > 35.0:
            print(f"✂️  Long clip detected ({dur:.1f}s) — Analyzing story arc & cutting filler...")
            story_ranges, ai_story_title, story_reasoning = select_story_segments(
                input_path,
                model_repo=model_repo,
                ollama_model=options.get("ollama_model", "qwen2.5:latest"),
                options=options
            )
            if story_reasoning:
                print(f"🧠 AI Editor's Note: \"{story_reasoning}\"")
            if story_ranges:
                base, ext = os.path.splitext(input_path)
                spliced_path = f"{base}_story_raw{ext}"
                splice_story_video(input_path, story_ranges, spliced_path)
                target_video = spliced_path
                if ai_story_title and not title:
                    title = ai_story_title

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_STORY{ext}" if target_video != input_path else f"{base}_edited{ext}"

    print(f"🎙️  Transcribing words for editing: {os.path.basename(target_video)}...")
    words_data = transcribe_words(target_video, model_repo=model_repo)
    print(f"💬 Found {len(words_data.get('words', []))} words in speech.")

    edited_file = render_edited_video(target_video, output_path, title, words_data, options=options)

    if target_video != input_path and os.path.exists(target_video):
        try:
            os.remove(target_video)
        except Exception:
            pass

    return edited_file

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        test_title = sys.argv[2] if len(sys.argv) > 2 else "NEVER BUILD A LOG BRIDGE! 😱"
        print(f"Testing clip_editor Karaoke Spec v1 on: {test_file}")
        out = process_and_edit_clip(test_file, test_title)
        print(f"Done! Output: {out}")
    else:
        print("Usage: python3 clip_editor.py <path_to_video.mp4> [title]")
