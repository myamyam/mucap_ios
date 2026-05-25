# Object Audio Caption Pipeline

This repository keeps source code only. Do not commit local media, generated outputs, model checkpoints, virtual environments, or API keys.

## Required Local Inputs

Place a clip in:

```text
input/<clip_id>.mp4
```

Create an object description JSON:

```json
{
  "instruments": {
    "violin": 3,
    "cello": 1
  }
}
```

Create `.env` from `.env.example` and set `OPENAI_API_KEY`.

## Main Pipeline

Recommended repository layout:

```text
scilab/
  samaudio/
  mucap/
```

`scripts/run_object_audio_pipeline.py` automatically looks for MuCap at `../mucap` first, then falls back to `../MuCap-main/MuCap-main` for the current local development layout.

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_object_audio_pipeline.py `
  --input input\<clip_id>.mp4 `
  --object-description object_description.json `
  --separation-model-id facebook/sam-audio-small-tv `
  --separation-device cpu `
  --run-mucap-captions `
  --render-captioned-video `
  --caption-position-mode fixed-by-scene
```

The wrapper runs:

```text
mp4 -> wav/frame extraction
GPT image object coloring
manifest creation
SAM-Audio separation
object coordinate CSV export
Realtime audio/object matching
MuCap caption JSON generation
scene cut detection
final captioned MP4 rendering
```

## Main Outputs

```text
outputs/video_assets/<clip_id>/
  manifest.json
  separated_audio/
  coords/object_coords.csv
  matching/time_object_audio_coords_full.csv
  matching/time_object_audio_coords_active_only.csv
  captions/mucap_output/
  scene_cuts.json
  final/captioned_video.mp4
```
