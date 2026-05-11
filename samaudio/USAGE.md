# SAM-Audio Runner Files

This folder contains code-only files for running SAM-Audio workflows. It intentionally excludes checkpoints, local datasets, virtual environments, and generated outputs.

## Structure

```text
samaudio/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ sam_audio/
└─ scripts/
   ├─ separate_raw_audio1_visual_objects.py
   ├─ frame_mask_coords.py
   └─ draw_bbox_overlays.py
```

## Common Commands

Separate object audio from color-highlighted frame folders:

```powershell
.\.venv\Scripts\python.exe -u scripts\separate_raw_audio1_visual_objects.py --audio audio_data/raw_audio1.wav --visual-dir raw_visual1_frames_colored --object-root raw_visual1_frames_coloredbyobject --output-dir outputs/raw_audio1_visual_objects_small --model-id facebook/sam-audio-small-tv --objects violin1 --reranking-candidates 1 --max-frames 2 --amp
```

Export per-frame mask coordinates:

```powershell
.\.venv\Scripts\python.exe scripts\frame_mask_coords.py --frame-dir raw_visual1_frames_coloredbyobject\violin1 --color red --csv outputs\violin1_red_coords.csv
```

Draw bbox and center overlays from a CSV:

```powershell
.\.venv\Scripts\python.exe scripts\draw_bbox_overlays.py --frames-dir raw_visual1_frames --coord-frames-dir raw_visual1_frames_colored --csv outputs_csv\time_object_audio_coords_active_only.csv --output-dir outputs_bbox_overlay_scaled
```

If the CSV coordinate image size already matches the source frame image size, omit `--coord-frames-dir`.
