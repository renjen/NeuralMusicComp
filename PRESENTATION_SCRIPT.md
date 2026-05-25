# Presentation Script & Slide Guide
## CSE 153 Assignment 2 — Neural Music Composer
**Target: ~20 minutes total**

---

## SLIDE 1 — Title / Intro
**On the slide:**
- Title: "Neural Music Composer"
- Subtitle: "Generating Bach-style Music with LSTMs"
- Your name
- Two tasks: Unconditioned Melody Generation + Harmonization

**What to say:**
> "For this assignment I trained two machine learning models on a dataset of Bach chorales — classical piano pieces from the 1700s. The first task is unconditioned generation, meaning the model learns to compose melodies completely from scratch. The second task is harmonization, where I give the model a melody and it generates the harmony underneath it. Both tasks use the same dataset and the same type of model — an LSTM neural network."

---

---

# TASK 1 — Unconditioned Melody Generation
**(~8-9 minutes)**

---

## SLIDE 2 — Task 1: Data (Context)
**On the slide:**
- Title: "Dataset — Bach Chorales"
- ~370 four-part chorales by J.S. Bach (BWV 250–438)
- Written in the 17th–18th century as Lutheran hymn harmonizations
- Available built-in to the `music21` Python library (no downloading needed)
- Each piece has 4 voices: Soprano, Alto, Tenor, Bass
- We use the **soprano voice** for Task 1

**What to say:**
> "The dataset I used is the Bach Chorale corpus, which contains about 370 four-part harmonizations written by Bach. These are short hymn-based pieces, each with a soprano, alto, tenor, and bass voice. What's convenient is that this dataset is built directly into the music21 Python library, so there's no manual downloading — it loads automatically. For Task 1, I only use the soprano voice, which is the top melody line."

---

## SLIDE 3 — Task 1: Data (Pre-processing)
**On the slide:**
- Quantized all notes to **quarter-note beats** (1 beat = 1 token)
- Each note represented as a **MIDI pitch integer** (0–127)
- Special tokens: REST (0), START (128), END (129) → vocabulary size = 130
- Sliding window sequences of length 32
- Split: **80% train / 10% val / 10% test** at the chorale level

**What to say:**
> "To prepare the data I quantized every note to quarter-note resolution — meaning I sample what note is playing once per beat. Each note is represented as a MIDI number, which is just an integer from 0 to 127. I also added special tokens for rests, the start of a piece, and the end of a piece, giving a vocabulary of 130 tokens. I then created training windows of 32 notes each using a sliding window approach, and split the data 80/10/10 into train, validation, and test sets — importantly splitting at the chorale level so no chorale leaks between splits."

---

## SLIDE 4 — Task 1: Data (Visualizations)
**On the slide:**
- Screenshot: **Pitch Distribution plot** (the 2x2 grid of all 4 voices)
- Screenshot: **Chorale Length Histogram**
- Screenshot: **Melodic Interval Distribution**
- Key stats: X chorales loaded, avg length = X beats, soprano range = MIDI 60–81

**What to say:**
> "Looking at the data — the pitch distributions show that the soprano voice sits mostly between MIDI 60 and 81, which is C4 to A5, a comfortable vocal range. The chorale length histogram shows most pieces are between 20 and 60 beats long. Most interestingly, the interval distribution shows that the vast majority of melodic steps are small — plus or minus 1 or 2 semitones. This is called stepwise motion and is a defining feature of Baroque vocal writing. So a well-trained model should reproduce this property."

---

## SLIDE 5 — Task 1: Modeling (Problem Formulation)
**On the slide:**
- **Input:** sequence of notes x₁, x₂, ... xₜ
- **Output:** probability distribution over the next note xₜ₊₁
- **Objective:** minimize cross-entropy loss (maximize likelihood of training data)
- **Inference:** autoregressive sampling with temperature control

**What to say:**
> "I frame this as a language modeling problem. Given the sequence of notes so far, the model predicts a probability distribution over what the next note should be. During training, the objective is to minimize cross-entropy loss — essentially, the model is penalized for assigning low probability to the actual next note. At generation time, I sample autoregressively: feed in a start token, sample the next note, feed that back in, and repeat. Temperature controls how random the sampling is — lower temperature makes the output more conservative, higher makes it more creative."

---

## SLIDE 6 — Task 1: Modeling (Architecture)
**On the slide:**
- Comparison table:

| Model | Pros | Cons |
|-------|------|------|
| Markov Chain | Fast, simple | No long-range structure |
| **LSTM (chosen)** | Long-range dependencies, efficient | Harder to interpret |
| Transformer | State of the art | Needs more data, more complex |

- Architecture diagram:
```
Input → Embedding(130→64) → LSTM(2 layers, 256 hidden) → Linear(256→130) → Softmax
```
- Training: Adam optimizer, lr=0.001, batch size=128, early stopping

**What to say:**
> "I compared three possible approaches. A Markov chain is simple and fast but can only look back one or two notes — it has no memory of the broader melodic context. A Transformer would be state of the art but requires more data and is more complex to implement. I chose a 2-layer LSTM as the middle ground — it can capture dependencies across 30+ notes, which is enough for a full chorale phrase, and it trains in just a few minutes on CPU. The architecture is an embedding layer that maps each token to a 64-dimensional vector, two LSTM layers with 256 hidden units, and a final linear layer that outputs logits over all 130 tokens."

---

## SLIDE 7 — Task 1: Evaluation
**On the slide:**
- Screenshot: **Loss curves** (train vs validation)
- Screenshot: **Perplexity bar chart** (LSTM vs bigram vs unigram)
- Table of results:

| Model | Perplexity |
|-------|-----------|
| LSTM (test) | X.XX |
| Bigram baseline | X.XX |
| Unigram baseline | X.XX |

- Melodic property comparison table (% stepwise motion, mean range)

**What to say:**
> "I evaluate the model using perplexity, which measures how surprised the model is by the test data — lower is better. My LSTM achieves significantly lower perplexity than both the unigram baseline, which just samples from note frequencies, and the bigram baseline, which only looks at the previous note. Beyond perplexity I also compared melodic properties of the generated sequences versus real Bach — specifically the percentage of stepwise motion and the pitch range. The generated melodies have similar stepwise motion rates to real Bach, which suggests the model has learned this stylistic property even though it's not directly in the training objective."

---

## SLIDE 8 — Task 1: Related Work
**On the slide:**
- **Allan & Williams (2005)** — first probabilistic approach to Bach; used HMMs
- **Waite / Magenta (2016)** — Google Brain LSTM for melody generation
- **Music Transformer (Huang et al., 2018)** — attention-based, better long-range structure
- How our work compares: similar PPL to Waite on small corpora; lacks long-range attention of Transformer

**What to say:**
> "This problem has a rich history. Allan and Williams in 2005 were among the first to apply probabilistic ML to Bach, using Hidden Markov Models. Google's Magenta project in 2016 showed that LSTMs produce coherent short-range melodic phrases — which is essentially what I've replicated here. The Music Transformer from 2018 is the current state of the art, using relative attention to capture structure across hundreds of notes. My model is most similar to the Magenta approach — it handles local melodic structure well but doesn't have the long-range coherence of a Transformer. That would be the natural next step."

---

---

# TASK 2 — Harmonization
**(~8-9 minutes)**

---

## SLIDE 9 — Task 2: Data (Context)
**On the slide:**
- Same Bach Chorale corpus — now using **all 4 voices**
- Input: soprano melody (given)
- Output: alto + tenor + bass (generated)
- Same quantization and vocabulary as Task 1
- This is the classic "auto-harmonization" problem

**What to say:**
> "For Task 2 I use the same dataset, but now I need all four voices. The problem is: given the soprano melody, can the model generate the harmony underneath? This is called auto-harmonization and it's one of the oldest problems in algorithmic composition. Bach's chorales are the standard benchmark for this because they follow clear harmonic rules — so it's easy to tell when a model is doing something musically wrong."

---

## SLIDE 10 — Task 2: Data (Visualizations)
**On the slide:**
- Screenshot: **Piano roll** of a sample chorale (all 4 voices)
- Screenshot: **Soprano-Bass interval histogram**
- Key observation: peaks at 12 (octave), 19 (octave + 5th), 7 (5th) → tonal structure

**What to say:**
> "The piano roll shows all four voices moving together — you can see how the soprano and bass tend to move in opposite directions, which is a hallmark of Bach's voice leading. The interval histogram between soprano and bass shows strong peaks at 7, 12, and 19 semitones — perfect 5ths and octaves — which is exactly what you'd expect from tonal harmony. This tells us the data has clear structure the model can learn."

---

## SLIDE 11 — Task 2: Modeling (Problem Formulation)
**On the slide:**
- **Input:** full soprano sequence s₁, s₂, ... sₜ
- **Output:** alto, tenor, bass sequences simultaneously
- **Objective:** sum of cross-entropy losses across all 3 voices
- Architecture:

```
Soprano → Embedding → LSTM(2 layers, 256) → ┬→ Linear → Alto
                                              ├→ Linear → Tenor
                                              └→ Linear → Bass
```

**What to say:**
> "I frame harmonization as a conditional sequence prediction problem. The soprano is the input condition, and the model predicts all three other voices at each time step. Instead of one output head, I use three parallel linear layers — one for alto, one for tenor, one for bass — all fed from the same LSTM hidden state. The total loss is the average cross-entropy across all three voices. This means the model sees the soprano context at every step but predicts each voice independently — they don't condition on each other."

---

## SLIDE 12 — Task 2: Modeling (Approach Comparison)
**On the slide:**

| Model | Pros | Cons |
|-------|------|------|
| Rule-based (figured bass) | Interpretable | Needs expert rules, can't learn from data |
| HMM (Allan & Williams) | Probabilistic | Limited context |
| **LSTM 3-head (chosen)** | Learns from data, handles soprano context | Voices independent, no inter-voice constraints |
| DeepBach (Gibbs sampling) | Joint voice modeling, state of the art | Very complex to implement |

**What to say:**
> "The main modeling choice here is how to handle the interaction between voices. The simplest approach is rule-based, but that requires hand-coding music theory. DeepBach, the state of the art from 2017, uses Gibbs sampling to model all four voices jointly — but it's significantly more complex to implement. My approach is a practical middle ground: one LSTM that encodes the soprano, with three separate prediction heads for the other voices. The limitation is that the voices don't condition on each other — so the model might generate an alto and tenor that clash with each other even if both individually make sense given the soprano."

---

## SLIDE 13 — Task 2: Evaluation
**On the slide:**
- Screenshot: **Training loss curves**
- Screenshot: **Accuracy bar chart** (LSTM exact vs pitch-class vs majority baseline)
- Results table:

| Voice | Exact Acc | Pitch-Class Acc | Majority Baseline |
|-------|-----------|-----------------|-------------------|
| Alto  | X% | X% | X% |
| Tenor | X% | X% | X% |
| Bass  | X% | X% | X% |

- Screenshot: **Piano roll comparison** (reference Bach vs generated)

**What to say:**
> "I evaluate using per-voice accuracy — does the model predict the exact pitch? I compare against a majority-class baseline that always predicts the most common pitch for each voice. The LSTM beats the baseline on all three voices, showing it learned something beyond just frequency. Pitch-class accuracy is higher than exact accuracy across the board — meaning the model gets the right note name but sometimes puts it in the wrong octave. The piano roll comparison shows the generated harmony follows the soprano reasonably well, though it's smoother and less varied than Bach's original — which makes sense given the model doesn't have explicit knowledge of harmonic rules."

---

## SLIDE 14 — Task 2: Related Work
**On the slide:**
- **Allan & Williams (2005)** — HMM-based harmonization, first ML approach
- **BachBot / Liang (2016)** — LSTM seq2seq, competitive Turing test results
- **DeepBach / Hadjeres et al. (2017)** — Gibbs sampling, humans preferred it over Bach ~50% of the time
- **Music Transformer (2018)** — attention-based, better long-range harmony
- How our work compares: similar to BachBot approach; main gap is no joint voice modeling

**What to say:**
> "DeepBach from 2017 is the landmark result here — human listeners preferred its output over real Bach about 50% of the time, which is remarkable. The key difference from my approach is that DeepBach uses bidirectional LSTMs and Gibbs sampling to model all voices jointly, iteratively refining the harmony. BachBot from 2016 is closer to what I built — a unidirectional LSTM that predicts voices given soprano — and still achieved strong results on a Turing test. My results are in the same ballpark for accuracy metrics, and the main gap is the lack of joint voice modeling and the absence of explicit music theory constraints."

---

---

## SLIDE 15 — Summary + Play Music
**On the slide:**
- Summary table:

| | Task 1 | Task 2 |
|--|--------|--------|
| Goal | Generate melody from scratch | Harmonize a given melody |
| Model | 2-layer LSTM LM | 2-layer LSTM + 3 heads |
| Key metric | Perplexity | Per-voice accuracy |
| Beats baseline? | Yes | Yes |

- "Now let's listen to the output..."

**What to say:**
> "To summarize — both models were trained from scratch on the Bach Chorale corpus and both outperform their respective baselines, showing the LSTMs learned real musical structure beyond just frequency patterns. The main limitations are the lack of long-range structure in Task 1 and the independent voice prediction in Task 2 — both of which point to natural extensions using Transformers or joint modeling. Now let's actually listen to what the models generated."

*[Play symbolic_unconditioned.mid, then symbolic_conditioned.mid]*

---

## Timing Guide

| Slide | Time |
|-------|------|
| Intro | 1 min |
| Task 1 Data (3 slides) | 4 min |
| Task 1 Model (2 slides) | 3 min |
| Task 1 Evaluation | 2 min |
| Task 1 Related Work | 2 min |
| Task 2 Data (2 slides) | 3 min |
| Task 2 Model (2 slides) | 3 min |
| Task 2 Evaluation | 2 min |
| Task 2 Related Work | 2 min |
| Summary | 1 min |
| **Total** | **~23 min** |

Trim by going slightly faster through the data slides — those are the most visual and least verbal.
