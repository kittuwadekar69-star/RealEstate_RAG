
from pathlib import Path
from faster_whisper import WhisperModel

BASE = Path(__file__).parent
VIDEO = BASE / "video.mp4"
OUTPUT = BASE / "data" / "video_transcript.txt"

if not VIDEO.exists():
    raise SystemExit(
        "Put your MP4 in this folder and name it video.mp4"
    )

print("Loading transcription model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

segments, info = model.transcribe(str(VIDEO), vad_filter=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    for segment in segments:
        line = (
            f"[{segment.start:.1f}s - {segment.end:.1f}s] "
            f"{segment.text.strip()}"
        )
        print(line)
        f.write(line + "\n")

print(f"\nTranscript saved to: {OUTPUT}")
