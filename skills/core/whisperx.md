# WhisperX / Transcription Skill

## When to Use

Use the `transcriber` tool whenever you need to convert speech to text from
audio or video files. This is the entry point for all transcript-dependent
workflows: subtitle generation, edit decisions based on spoken content, and
scene analysis from dialogue.

## Tool

| Tool | Capability |
|------|-----------|
| `transcriber` | Speech-to-text with word timestamps, language detection, optional diarization |
| `local_whisper_stt` | Local spark-41db gateway transcription with segments + word timestamps, no diarization |

## How It Works

1. **Default local path:** for normal single-speaker / subtitle workflows, `transcriber` now prefers the live local gateway-backed provider `local_whisper_stt` when it is available at `http://127.0.0.1:30008`.
2. **Structured transcription:** the local gateway returns `text`, `segments`, and `word_timestamps`, which is the shape `subtitle_gen` and transcript-driven QA expect.
3. **Diarization (optional):** when `diarize=True`, `transcriber` falls back to the in-process WhisperX/faster-whisper path so speaker labeling can still happen. Diarization requires `whisperx` and usually `HF_TOKEN` for pyannote-backed speaker assignment.

## Model Size Guide

| Model | RAM | Speed (CPU) | Quality | When to Use |
|-------|-----|-------------|---------|-------------|
| `tiny` | ~1 GB | ~10x real-time | Low | Quick drafts, iteration |
| `base` | ~1 GB | ~5x real-time | Good | Default for development |
| `small` | ~2 GB | ~3x real-time | Better | Short content |
| `medium` | ~5 GB | ~1.5x real-time | High | Important content |
| `large-v3` | ~10 GB | ~0.5x real-time | Best | Final production |

## Key Patterns

### Choosing When to Diarize

- **Single speaker (talking head):** Skip diarization — it adds latency with no benefit.
- **Multiple speakers (interview, podcast):** Enable diarization to label who said what.
- **Diarization requires** `whisperx` and `HF_TOKEN`. If unavailable, the tool proceeds without speaker labels.

### Word Timestamps for Subtitles

The transcriber produces word-level timestamps with confidence scores. The `subtitle_gen` tool consumes these directly:

```
word_timestamps: [
  {"word": "Hello", "start": 0.5, "end": 0.8, "probability": 0.95},
  {"word": "world", "start": 0.9, "end": 1.2, "probability": 0.92},
  ...
]
```

### Language Detection

- Pass `language: null` to auto-detect (adds ~1s overhead).
- Pass an explicit ISO 639-1 code (`en`, `es`, `ja`, etc.) when you know the language.

## Quality Checklist

- [ ] Transcript text is accurate (spot-check 3-5 segments)
- [ ] Word timestamps align with actual speech when played back
- [ ] No missing segments or large gaps in the transcript
- [ ] Language was correctly detected (if auto)
- [ ] Speaker labels are correct (if diarization was used)
