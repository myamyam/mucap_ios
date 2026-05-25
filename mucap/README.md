# μCap (muCap)

**μCap** is an automatic instrumental music captioning system designed to improve music accessibility for Deaf and Hard-of-Hearing (DHH) individuals. Unlike conventional captions that reduce music to generic labels (e.g., “[instrumental music]”), μCap generates **time-aligned, sound-mimetic textual captions** that phonetically reflect musical elements such as rhythm, pitch, dynamics, and timbre. Grounded in user studies with DHH participants and expert discussions spanning linguistics, music, and audio engineering, μCap integrates audio feature extraction with a **Retrieval-Augmented Generation (RAG)** pipeline to produce expressive, non-lexical captions enhanced with simple visual cues. User evaluations demonstrate that μCap significantly improves music appreciation, immersion, and perceived access to acoustic detail, highlighting the potential of text-based representations as an inclusive medium for instrumental music.

![μCap Logo](images/mucap.png)

## 📦 1. Installation

```bash
conda create -n mucap python=3.11
conda activate mucap
pip install -r requirements.txt
```

## 📁 2. Dataset Preparation

### Dataset Directory Structure
Download datasets (e.g., [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), [MER](https://mir.dei.uc.pt/downloads.html)) and place them in the `dataset/` directory.

```
dataset/
├─ GTZAN/
│   ├─ blues/
│   │   ├─ blues.00000.wav
│   │   ├─ blues.00001.wav
│   │   └─ ...
├─ MER/
│   ├─ ...
```

### 🎙️ Vocal Separation
For audio tracks that contain vocals, we convert them into instrumental versions by separating the vocal component using spleeter and generating a corresponding vocal-removed WAV file.

```bash
bash script/process_vocal.GTZAN.sh
bash script/process_vocal.MER.sh
```

### ✂️ Segment Splitting
Since our work uses 14-second audio segments, longer audio files (e.g., 30 seconds) are divided into shorter segments according to the defined duration.
```bash
python split_wav.py
```

### 🎵 Audio Feature Extraction
We extract various features from the audio, including file_name, genre, dataset, sample_rate, length, volume, pitch, tempo, attack, envelope, onset_strength, resonance, notes, beat_frames, top5_instruments, and file metadata. These features are obtained using the `essentia` library.
```bash
python music_features/extract_dataset.py
```

## 🤖 3. LLM Music Description Generation
To use the LLM, the API key must be registered as an environment variable. The API key can be obtained [OpenAI API](https://platform.openai.com/).
```bash
cp .env.example .env
# Edit .env and set your OpenAI API key:
OPENAI_API_KEY=sk-proj-xxxx
```
When running the script uses the `gpt-4o-audio-preview` model to generate descriptive files for each audio segment. At this stage, captions are not produced; instead, the model generates perceptual and auditory descriptions of the music.

```bash
python collect_annotation.py
```

## 📚 4. Build RAG Database
Using the `langchain` package, we convert the music features, model-generated descriptions, and human annotations into text representations and embed them into a vector database.
The embedding model used for this process is `text-embedding-3-large`.

```bash
python mucap/rag/json2ragdb.py
```

## 🎬 5. Run μCap Caption Generation
When you run the command below, captions are generated for all WAV files located in the `dataset` folder. In this process, the system first extracts audio features, then queries the RAG database using those features, and finally augments the caption generation using the retrieved information.

```bash
bash script/mucap_batch.sh
```

The `--use_rag [true|false]` and `--use_design_prompt [true|false]` flags allow you to enable or disable the RAG module and the design prompts, respectively.

### Output Example

```
output/
├─ GTZAN/
│   ├─ blues/
│   │   ├─ blues.00062.novocal_1-_af_rag.json
│   │   ├─ blues.00062.novocal_1-_af_rag.txt
│   │   ├─ blues.00062.novocal_1-_af_rag.png
│   │   └─ blues.00062.novocal_1-_af_rag.mp4
```

If you wish to generate captions for a dataset that is not part of the original training sets, simply create a new folder (e.g., testset) inside the dataset directory, organize the audio files using the same folder structure as the other datasets, and run the pipeline as usual.


- `*.json`: μCap request/response logs
- `*.txt`: final tokenized captions
- `*.png`: music spectrogram
- `*.mp4`: final synchronized caption video
