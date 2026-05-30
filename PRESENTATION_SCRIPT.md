# Presentation Script — Neural Music Comp
**22 slides | ~20 minutes | 3 speakers**

---

## Speaker Split

| Speaker | Slides | Section |
|---------|--------|---------|
| **Renee** | 1–8 | Title, Overview, Task 1 Data (easy — all visuals + stats) |
| **Aditi** | 9–12 | Task 1 Modeling, Evaluation, Related Work |
| **Chris** | 13–22 | All of Task 2 + Summary |

---

---

# RENEE'S PART (~7 minutes)

---

## Slide 1 — Title
**Say:**
> "Hi everyone — our project is called Neural Music Composer. We trained two machine learning models on a dataset of classical music by Bach to generate new music. I'm Renee, and I'll be walking through our overview and our first task's data. Aditi will cover the modeling and results for Task 1, and Chris will cover Task 2."

---

## Slide 2 — Overview
**Say:**
> "The big idea is this: we used a dataset called the Bach Chorale corpus, which is a collection of about 370 classical music pieces that comes built into a Python library called music21. We used it to train two different neural network models.

> Task 1 is unconditioned generation — the model looks at previous notes and learns to predict what note should come next, then uses that to generate a brand new melody from scratch.

> Task 2 is harmonization — you give it a melody, and it generates the other three voices underneath to create a full four-part piece.

> Both tasks use the same core setup: notes are represented as MIDI numbers, we quantize to quarter-note beats, and we use special start, end, and rest tokens."

---

## Slide 3 — Task 1 Title Slide
**Say:**
> "Let's dive into Task 1 — Unconditioned Melody Generation. This falls under what the assignment calls symbolic unconditioned generation — symbolic because the output is notes, not audio, and unconditioned because there's no input, the model just generates freely."

---

## Slide 4 — Dataset: Bach Chorales
**Say:**
> "The dataset is the Bach Chorale corpus — roughly 370 four-part harmonizations written by J.S. Bach between the 1700s. Each piece is a short Lutheran hymn arranged for four voices: soprano, alto, tenor, and bass. What's really convenient is that this entire dataset is embedded inside the music21 Python library, so we didn't need to download anything — it loads automatically when you run the code.

> For Task 1, we only use the soprano voice, which is the top melody line."

---

## Slide 5 — Data Pre-processing
**Say:**
> "Here's how we prepared the data. There are four steps.

> First, we quantize — meaning we sample one note per quarter-note beat. So instead of capturing every tiny rhythmic detail, we simplify to one token per beat.

> Second, we encode each note as a MIDI pitch integer — a number from 0 to 127 — and add three special tokens: REST for silences, START to mark the beginning of a piece, and END to mark the end. This gives us a vocabulary of 130 tokens total.

> Third, we create sliding windows of 32 notes. So for each sequence of 32 consecutive notes, we ask the model to predict the next one.

> Fourth, we split the data at the chorale level — 80% for training, 10% for validation, and 10% for testing. Splitting at the chorale level is important so that notes from the same piece don't end up in both train and test.

> That gives us 16,903 training windows, 1,816 validation windows, and 2,289 test windows."

---

## Slide 6 — Exploratory Data Analysis (Stats Table)
**Say:**
> "Let's look at what's actually in the data. Each voice has around 20,000 notes across all 368 chorales. The soprano, which is what we're training on for Task 1, uses 23 unique pitches and has a mean MIDI pitch of about 70, which is around B4 — a pretty comfortable singing range. The pitch range is MIDI 57 to 81, which is A3 to A5.

> Interesting to note: the bass has the widest range — from MIDI 36 all the way to 64 — which is typical in choral writing where the bass covers a lot of harmonic ground.

> The average chorale is 55 beats long, and soprano melodies mostly move in small steps, which is characteristic of Baroque vocal writing."

---

## Slide 7 — Pitch Distribution (Charts)
**Say:**
> "These four histograms show the pitch distributions for each voice. You can see that each voice occupies a distinct range — soprano clusters around the 60s and 70s, alto a little lower, tenor lower still, and bass down in the 40s and 50s. The distributions look roughly bell-shaped, which reflects the natural constraints of singing — voices stay close to their center of range and don't jump around randomly.

> The dashed vertical lines mark octave boundaries — you can see how cleanly each voice fits within its expected range."

---

## Slide 8 — Chorale Length & Melodic Interval (Charts)
**Say:**
> "Two more data visualizations. On the left is the distribution of chorale lengths — most pieces are between 30 and 60 quarter-note beats, with a mean of 55.3. The red dashed line marks that mean. A few outliers go up to 200 beats.

> On the right is the melodic interval distribution for the soprano — meaning, how many semitones does the melody jump between consecutive notes. You can see a huge spike at 0 — a lot of repeated notes — and then smaller steps of plus or minus 1 and 2 semitones dominate. Bigger jumps are rare. This stepwise motion is a defining feature of Bach's melodic style, and it's something we can actually measure in our generated melodies to see if the model learned it."

---

---

# ADITI'S PART (~6 minutes)

---

## Slide 9 — Model Formulation
**Say:**
> "Now let's talk about how we turned this into a machine learning problem. The input is a sequence of soprano tokens — x1, x2, up to xt. The model predicts p of x at t+1 given all previous tokens — in other words, a probability distribution over the 130 possible next notes. The objective during training is to minimize cross-entropy loss, which means we penalize the model when it assigns low probability to the actual next note. At generation time, we sample autoregressively — we feed in a start token, sample the next note, feed that back in, and repeat until we get an end token."

---

## Slide 10 — Modeling: Architecture
**Say:**
> "We compared three types of models. A Markov chain is fast and simple but can only look back one note at a time — it has no real memory. A Transformer is state of the art for this kind of sequence modeling but requires more data and is more complex to implement. We chose an LSTM, which is a good middle ground — it can remember context across 30 or more notes, which is enough for a full phrase, and it trains in minutes on a CPU.

> The architecture is: an embedding layer that maps each of the 130 tokens to a 64-dimensional vector, then two LSTM layers with 256 hidden units and dropout of 0.3 to prevent overfitting, then a linear layer that outputs 130 logits for the next note.

> We trained with the Adam optimizer, learning rate 0.001, batch size 128, and used early stopping to prevent overfitting."

---

## Slide 11 — Evaluation (Task 1)
**Say:**
> "We evaluate using perplexity, which measures how surprised the model is by unseen notes — lower perplexity means the model has learned the distribution better. We compare against two baselines: a unigram model that just samples notes based on how often they appear in training data, and a bigram model that looks at the previous single note to predict the next.

> Our LSTM achieves a test perplexity of 5.3. The bigram gets 8.0, and the unigram gets 16.1. So our model is significantly better than both baselines — it's learned real sequential structure.

> We also measured melodic properties of our generated output. The mean interval is 1.41 semitones, and 89.4% of consecutive notes are stepwise motion — which closely matches what we measured in the real Bach data on slide 8. So the model hasn't just memorized low perplexity — it's actually generating music with the right stylistic properties."

---

## Slide 12 — Related Work (Task 1)
**Say:**
> "For context, here's how our work fits into the research landscape. Allan and Williams in 2005 were among the first to apply probabilistic ML to Bach, using Hidden Markov Models. Google's Magenta project in 2016 showed that LSTMs can generate coherent short-range melodic phrases — which is essentially what we replicated here. The Music Transformer from 2018 is the current state of the art, using attention to model structure across hundreds of notes. And MIDI-VAE from 2018 adds a latent space for more controlled generation.

> Our model is most comparable to the Magenta LSTM — strong on local note-to-note patterns, but the main gap is that it doesn't reliably produce longer phrase-level repetition, which would require something like the Music Transformer's attention mechanism."

---

---

# CHRIS'S PART (~7 minutes)

---

## Slide 13 — Task 2 Title Slide
**Say:**
> "Now moving on to Task 2 — Harmonization. This is what the assignment calls symbolic conditioned generation. Conditioned means we're given an input — the melody — and we generate an output conditioned on that input. The goal is to take just the top melody line and have the model fill in the harmony underneath."

---

## Slide 14 — Task 2 Data
**Say:**
> "We reuse the exact same Bach Chorale dataset from Task 1 — same quantization, same vocabulary, same train/val/test split. The difference is that now we use all four voices instead of just soprano.

> For the exploratory analysis, we plotted a Soprano-Bass interval histogram, which shows how far apart the soprano and bass tend to be at each moment. You can see peaks at 7 semitones — a perfect 5th — and at 12 and 19 — an octave and an octave plus a 5th. These are the most consonant intervals in Western music, so this histogram is basically confirming that the dataset has real tonal structure the model can learn from.

> We also have a piano roll visualization — that's the next slide."

---

## Slide 15 — Piano Roll
**Say:**
> "This is the piano roll for one chorale — each of the four colored lines is a different voice, plotted against time in quarter-note beats. You can see how the soprano stays at the top, alto and tenor in the middle, and bass at the bottom. You can also see how all four voices move roughly together — they change notes at similar times — which is the block-chord style typical of Bach chorales. This gives us a visual sense of what the model needs to learn."

---

## Slide 16 — Task 2 Modeling
**Say:**
> "For Task 2, the inputs are the full soprano sequence, and the outputs are alto, tenor, and bass simultaneously. We optimize by taking the average of cross-entropy losses across all three voices.

> We considered four approaches. Rule-based figured bass is interpretable but requires hand-coding music theory. An HMM is probabilistic but has limited context. DeepBach, which is the state of the art from 2017, uses Gibbs sampling to model all four voices jointly — but it's significantly more complex to implement. We chose an LSTM because it learns directly from data and handles the soprano context well.

> The architecture is the same LSTM backbone as Task 1, but instead of one output head, we have three parallel linear layers — one each for alto, tenor, and bass. The main limitation is that each voice is predicted independently — they don't condition on each other — so the model might not enforce rules like no parallel fifths between voices."

---

## Slide 17 — Evaluation: Training Curves
**Say:**
> "Here are the training curves for the harmonizer. Both training loss and validation loss drop steeply in the first few epochs and then level off. The fact that validation loss tracks training loss closely — without diverging — tells us the model isn't overfitting. We used early stopping which kicked in around epoch 27."

---

## Slide 18 — Evaluation: Accuracy Table
**Say:**
> "For evaluation, we measure per-voice accuracy — does the model predict the exact pitch? We compare against a majority-class baseline that always predicts the single most common pitch for each voice.

> The LSTM outperforms the baseline for all three voices. For alto, the LSTM gets about 32% exact accuracy versus the majority baseline of 15%. For tenor it's 27% versus 17%, and for bass 21% versus 14%.

> The pitch-class accuracy — which checks if the model got the right note name, ignoring octave — is similar to exact accuracy. This means when the model is wrong, it's not just getting the wrong octave — it's actually getting a different note, which suggests the harmony isn't trivially easy."

---

## Slide 19 — Evaluation: Piano Roll Comparison
**Say:**
> "This is perhaps the most intuitive evaluation — a direct visual comparison between Bach's original harmonization on top and our generated harmonization on the bottom. The soprano line in blue is the same in both — that's the input. The generated alto, tenor, and bass on the bottom follow the soprano reasonably well. The shapes look similar — you can see the voices mostly staying in their correct ranges and moving in similar directions to the reference. The generated version is smoother and less varied, which makes sense — the model has learned average behavior from the training set, not the specific harmonic choices Bach would make."

---

## Slide 20 — Related Work (Task 2)
**Say:**
> "For Task 2, the key benchmark is DeepBach from Hadjeres et al. in 2017 — their model was so convincing that human listeners preferred its output over real Bach about 50% of the time. The key difference from our approach is that DeepBach uses bidirectional LSTMs and Gibbs sampling to jointly model all four voices, iteratively refining the harmony. BachBot from 2016 is actually the closest to what we built — a unidirectional LSTM — and it still performed well on a Turing test. Our results are in a similar range. The main gap between our work and the state of the art is the lack of joint voice modeling and explicit harmonic constraints."

---

## Slide 21 — Summary
**Say:**
> "To wrap up — both models were trained from scratch on the Bach Chorales and both beat their baselines. Task 1 gets a perplexity of 5.3 versus 8.0 for bigram. Task 2 gets accuracy roughly double the majority-class baseline across all three voices. The main limitations are long-range structure for Task 1 and independent voice prediction for Task 2 — both natural extensions toward Transformers or joint modeling."

---

## Slide 22 — Thank You
**Say:**
> "Thanks — now let's listen to what the models actually generated."

*[Play symbolic_unconditioned.mid — the generated melody]*
*[Play symbolic_conditioned.mid — the 4-voice harmonization]*

---

---

## Quick Reference: Key Numbers to Remember

| Fact | Value |
|------|-------|
| Chorales in dataset | 368 |
| Avg chorale length | 55.3 beats |
| Vocabulary size | 130 tokens |
| Training windows (Task 1) | 16,903 |
| LSTM test perplexity | 5.3 |
| Bigram baseline perplexity | 8.0 |
| Stepwise motion in generated melodies | 89.4% |
| Alto exact accuracy | 31.9% |
| Alto majority baseline | 15.2% |
