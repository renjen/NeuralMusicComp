# Neural Music Composer

Two LSTM models trained on the **Bach Chorale dataset** to generate original music — one generates melodies from scratch, the other harmonizes a given melody into a full four-part chorale.

Built with PyTorch and music21. No pre-trained weights — everything is trained from scratch.

---

## What This Project Does

| Task | What it does | Output |
|------|-------------|--------|
| **Unconditioned Melody Generation** | LSTM learns Bach melody patterns and generates new ones from scratch | `output/symbolic_unconditioned.mid` |
| **Harmonization** | LSTM takes a melody (soprano) as input and generates alto, tenor, and bass voices | `output/symbolic_conditioned.mid` |

---

## Project Structure

```
NeuralMusicComposer/
├── notebooks/
│   └── workbook.ipynb        # Full notebook — data, models, evaluation, results
├── output/
│   ├── symbolic_unconditioned.mid  # Generated melody (Task 1)
│   └── symbolic_conditioned.mid    # Generated 4-voice chorale (Task 2)
├── scripts/
│   └── create_notebook.py    # Regenerates workbook.ipynb from scratch
├── .gitignore
└── README.md
```

---

## The Data

**No downloading required.** The dataset is built into the `music21` Python library.

- **Dataset:** Bach Chorales (BWV 250–438)
- **Source:** Embedded in `music21`'s internal corpus — loads automatically
- **Format:** MusicXML → parsed into MIDI pitch integers at quarter-note resolution
- **Size:** ~370 chorales, ~40 beats each, 4 voices per chorale
- **Split:** 80% train / 10% validation / 10% test (split at the chorale level)

When the notebook runs `corpus.parse(path)`, music21 reads from its own files — no internet needed after the library is installed.

---

## Setup

### 1. Install dependencies

```bash
pip install music21 pretty_midi mido torch numpy pandas matplotlib seaborn nbformat jupyter
```

All packages are standard. `torch` is PyTorch (CPU is fine — training takes a few minutes).

### 2. (Optional) Regenerate the notebook

If you want to rebuild `workbook.ipynb` from the creation script:

```bash
python3 scripts/create_notebook.py
```

This rewrites `notebooks/workbook.ipynb` with clean cells and no outputs.

---

## How to Run

### Option A — Run the full notebook interactively (recommended)

```bash
jupyter notebook notebooks/workbook.ipynb
```

Then in Jupyter: **Kernel → Restart & Run All**

This will:
1. Load the Bach Chorale data from music21
2. Show exploratory plots (pitch distributions, chorale lengths, etc.)
3. Train the Task 1 melody LSTM (~3–5 min on CPU)
4. Evaluate it and compare to baselines
5. Generate a new melody → save as `symbolic_unconditioned.mid`
6. Train the Task 2 harmonization LSTM (~3–5 min on CPU)
7. Evaluate it and compare to baselines
8. Generate a harmonized chorale → save as `symbolic_conditioned.mid`

### Option B — Run headlessly via command line

```bash
jupyter nbconvert --to notebook --execute notebooks/workbook.ipynb \
    --output notebooks/workbook_executed.ipynb \
    --ExecutePreprocessor.timeout=600
```

---

## What Each Notebook Section Contains

### Shared Data Pipeline
- Loads all Bach chorales from `music21`
- Quantizes notes to quarter-note beats
- Computes dataset statistics (note counts, pitch ranges, etc.)
- 4 visualizations: per-voice pitch distributions, chorale length histogram, melodic interval distribution, pitch-class frequency

### Task 1 — Unconditioned Melody Generation

| Section | Contents |
|---------|----------|
| **1.1 Data** | Soprano voice extraction, tokenization, sliding-window sequences |
| **1.2 Modeling** | 2-layer LSTM (64-dim embedding → 256-dim hidden → 130-class softmax). Trained with cross-entropy loss, Adam optimizer, early stopping |
| **1.3 Evaluation** | Perplexity on test set vs. unigram and bigram baselines. Melodic property comparison (% stepwise motion, pitch range) |
| **1.4 Related Work** | Allan & Williams 2005, Waite/Magenta 2016, Music Transformer 2018 |

**Model architecture:**
```
Input tokens → Embedding(130, 64) → LSTM(2 layers, 256 hidden) → Linear(256, 130)
```

### Task 2 — Harmonization (Conditioned Generation)

| Section | Contents |
|---------|----------|
| **2.1 Data** | All 4 voices extracted and aligned, piano-roll visualization, voice-interval histogram |
| **2.2 Modeling** | Same LSTM backbone but with 3 output heads (one per voice: alto, tenor, bass). Trained with summed cross-entropy across all 3 voices |
| **2.3 Evaluation** | Per-voice exact-pitch accuracy and pitch-class accuracy vs. majority-class baseline. Piano-roll comparison: reference Bach vs. generated |
| **2.4 Related Work** | Allan & Williams 2005, DeepBach 2017, BachBot 2016, Music Transformer 2018 |

**Model architecture:**
```
Soprano → Embedding(130, 64) → LSTM(2 layers, 256 hidden)
                                      ├→ Linear(256, 130) → Alto
                                      ├→ Linear(256, 130) → Tenor
                                      └→ Linear(256, 130) → Bass
```

---

## Listening to the Output

Both `.mid` files can be opened with:
- **GarageBand** (Mac) — drag and drop
- **MuseScore** (free) — full score view
- **VLC** — plays MIDI directly
- Any DAW (Logic, Ableton, etc.)

`output/symbolic_unconditioned.mid` — single-voice melody generated from scratch
`output/symbolic_conditioned.mid` — 4-voice chorale with generated harmony over a real Bach soprano

---

## Submission Checklist

- [ ] `workbook.html` — exported notebook (already generated)
- [ ] `symbolic_unconditioned.mid` — Task 1 music (already generated)
- [ ] `symbolic_conditioned.mid` — Task 2 music (already generated)
- [ ] `video_url.txt` — one line: Google Drive or YouTube link to ~20 min video
- [ ] Record and upload the video presentation

---

## Troubleshooting

**Training is slow** — the notebook uses CPU by default. ~5 min total is normal. If you have a GPU, PyTorch will use it automatically (`device = cuda`).

**`music21` can't find chorales** — run `python3 -c "from music21 import corpus; print(corpus.getComposer('bach')[:3])"` — if it returns a list of paths, music21 is working correctly.

**MIDI files sound wrong** — this is expected for ML-generated music. The model learned statistical patterns, not music theory rules. Some generated pieces will sound better than others.

**Notebook errors after re-running** — make sure to **Restart & Run All** (not just re-run individual cells) since variables depend on earlier cells.
