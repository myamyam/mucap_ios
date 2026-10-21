from pathlib import Path
from multimodal_caption.mobile_pipeline import transcribe

result = transcribe(Path("kung.wav").read_bytes())

for cue in result["captions"]:
    print(f'[{cue["time"]:.2f}] {cue["text"]}')
