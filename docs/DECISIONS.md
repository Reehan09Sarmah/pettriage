# PetTriage — Architecture Decision Records (ADRs)

> Decisions are logged here as they are made — not pre-planned.
> This answers: "Why did we build it this way?"

---

## ADR-001: Two Separate Classifiers
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
Two classifiers — one for Condition, one for Urgency — instead of a single multi-output model.

**Reason:**
- Urgency can be correct even when Condition is wrong
- Feedback can be collected and used separately for each
- Easier to retrain, version, and monitor independently

---

## ADR-002: Soft Cascade Architecture
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
Urgency classifier receives Model A's **full probability distribution** (soft labels), not the top predicted condition (hard label).

**Reason:**
Hard labels cause error propagation. If Model A is 40% UTI / 35% Kidney Stones / 25% other, that uncertainty is meaningful signal. Passing only "UTI" discards it.

**Example:**
```
Hard label approach → Model A: "Stomach Upset" → Model B: "Monitor" ← WRONG, dangerous
Soft label approach → Model A: {Internal Bleeding: 0.35, ...} → Model B: "Emergency" ← correct
```

---

## ADR-003: scikit-learn for Phase 1
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
Use scikit-learn (RandomForest / LogisticRegression) for initial model, not deep learning.

**Reason:**
- Dataset will be small initially
- Deep learning needs volume to outperform classical ML
- scikit-learn models are interpretable — important for health applications
- Easy to swap out later without touching the rest of the system

---

## ADR-004: Feedback Trust Asymmetry
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
Urgency feedback = high trust. Condition feedback = low trust.

**Reason:**
Owners know after the fact if it was an emergency. They do not reliably know the medical condition — that requires a vet. Using owner-supplied condition labels as ground truth introduces label noise.

---

## ADR-005: MLflow for Experiment Tracking
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
MLflow running locally for Phase 1.

**Reason:**
Free, open source, no account needed. Tracks params, metrics, and model artifacts. Migrating to a remote server later requires zero code changes.

---

## ADR-006: uv as Package Manager
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
`uv` instead of `pip` for all dependency management.

**Reason:**
Faster, modern, compatible with standard `requirements.txt`. Already installed.

---

## ADR-007: Synthetic Data for Phase 1
**Date:** 2026-08-26
**Status:** Accepted

**Decision:**
Use LLM-generated synthetic data as the sole training dataset for Phase 1. Real Kaggle data used as seed/reference only.

**Reason:**
- Real datasets lack Urgency labels
- Real datasets are thin (<5 rows for many categories)
- Synthetic data lets us control class balance and label quality
- MLOps pipeline is identical regardless of data source
- Phase 2 will retrain with real data + user feedback

---

## ADR-008: Seed Data Not Merged into Training
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
The 147-row seed dataset (`dogs_cats_labeled.csv`) is NOT merged into `training_data.csv`.

**Reason:**
The seed data has no `Urgency` column — the primary label for Model B. Merging it would require either dropping it (losing rows) or imputing urgency labels (introducing noise). Synthetic data already provides well-labeled, balanced training examples.

---

## ADR-009: 11 Disease Categories
**Date:** 2026-08-26
**Status:** Accepted

**Decision:**
Group all conditions into 11 clinical categories instead of predicting specific disease names.

**Categories:**
Respiratory, Gastrointestinal, Viral Systemic, Bacterial/Parasitic,
Skin/Fungal, Musculoskeletal, Eye/Ear, Renal/Urinary,
Endocrine/Metabolic, Dental/Oral, Trauma/Poisoning

**Reason:**
- 139 specific diseases → 3 rows per disease average (not learnable)
- Categories are more actionable for pet owners ("Respiratory" vs "Feline Herpesvirus")
- 11 categories with 200 rows each = learnable, balanced training set
- LLM generation is cleaner per category than per specific disease

---

## ADR-010: No PetMD Scraping
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
Do not scrape PetMD or similar veterinary websites.

**Reason:**
- ToS prohibits automated scraping for commercial purposes
- LLMs (Claude, Gemini) have already ingested PetMD content during training
- Generating via LLM is legally equivalent, faster, and produces cleaner labeled data
- The knowledge is the same; only the access method differs

---

## ADR-011: Stratified Split on Combined Category+Urgency Key
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
Split `training_data.csv` into train/val/test (70/15/15) stratified on a combined
`Disease_Category + "_" + Urgency` key, not a plain random split or single-column stratify.

**Reason:**
Category and Urgency are entangled (e.g. Trauma/Poisoning is 65% Emergency, Skin/Fungal is
5%). Stratifying on Urgency alone could still cluster all of one category into a single
split. Some combined groups are as small as 10 rows out of 2,200 — a random or
single-column-stratified split risked near-zero representation of rare combinations in
val/test.

---

## ADR-012: Multi-Hot Symptom Encoding Replaces Positional Slots + Legacy Flags
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
Replace the 4 positional `Symptom_1-4` columns and the 9 legacy Yes/No flag columns
(`Vomiting`, `Diarrhea`, ...) with a single set of 31 binary `Symptom_<name>` columns (one
per distinct symptom string found in the data), set to 1 if that symptom appears anywhere in
a row's 4 slots.

**Reason:**
- The 9 legacy flags covered only 9 of 31 distinct symptom strings actually present in
  `Symptom_1-4` — symptoms like `Fever`, `Lethargy`, `Seizures`, `Weight Loss` had no column.
- Positional slots implied a false ordering (no real "1st" vs "4th" symptom).
- Cross-checked derived columns against the 9 legacy flags: 0 mismatches, confirming they're
  a redundant subset — keeping both would add duplicated, collinear signal with no benefit.
- Added `Symptom_Count` (total symptoms reported) as a candidate severity feature.

---

## ADR-013: Duration Parsed to Continuous Days, Not Ordinal Categories
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
Parse the free-text `Duration` column (`"2 days"`, `"3 weeks"`, `"A few hours"`, ...) into a
single continuous `Duration_Days` numeric feature instead of ordinal/categorical encoding of
the 18 distinct strings.

**Reason:**
Ordinal/categorical encoding would treat each of the 18 strings as a label with no
inherent magnitude relationship. Parsing to days preserves the true "more time elapsed =
larger number" relationship, letting a model learn a smooth risk-vs-duration signal instead
of 18 unrelated categories.

---

## ADR-014: Breed Dropped from Feature Set; Gender Kept
**Date:** 2026-08-29
**Status:** Accepted

**Decision:**
Drop `Breed` (25 distinct values) from the model features entirely — not one-hot encoded, not
kept in any form. `Gender` is kept and one-hot encoded alongside `Animal_Type`.

**Reason:**
- Tested with Cramér's V (association strength, 0-1): `Breed` scored ~0.12 against both
  Disease_Category and Urgency — barely above negligible. `Animal_Type`, with only 2 possible
  values vs. Breed's 25, scored *higher* (~0.17) against Disease_Category — a cleaner signal
  with far less complexity. The p-values for Breed were technically "significant" but with
  2,200 rows spread across 25 categories that's expected from noise alone; effect size
  (Cramér's V) is what matters, not just detectability.
- Structural generalization problem independent of the weak signal: the 25 breeds present are
  only what this synthetic batch happened to generate. Real input will include breeds never
  seen in training; even with `handle_unknown='ignore'` preventing a crash, an unseen breed
  becomes an all-zero one-hot row — meaning Breed could only ever help for these exact 25
  breeds, never generalizing further.
- `Gender` kept despite not being tested the same way — engineer's call based on direct
  clinical reasoning (e.g. ovarian conditions only apply to female animals), independent of
  what association tests on this synthetic sample show. Only 2 possible values, so no
  generalization concern like Breed.

**Effect:** `train.csv`/`val.csv`/`test.csv` reduced from 70 to 45 columns.

---

## ADR-015: Prediction Output — Top Prediction + Close Alternatives, Not Fixed Top-N
**Date:** 2026-08-29
**Status:** Accepted (not yet implemented — applies to `predict.py`/API layer, not built yet)

**Decision:**
When Model A returns its probability distribution over the 11 disease categories, the
API/output layer will show the top prediction always, plus any other category whose
probability is within a set margin of the top score (e.g. within ~15 percentage points),
capped at a small number of alternatives. Not a hard rule of "always show 1" or "always show
top 3" — the number of categories shown depends on how confident the model actually is on
that specific input.

**Reason:**
- If the top two probabilities are close (e.g. 46% vs 41%), presenting only the top one as
  "the answer" overstates the model's actual confidence and risks hiding a real alternative —
  a meaningful risk in a health-triage context.
- If the model is genuinely confident (e.g. 89% vs everything else under 3%), listing
  low-probability alternatives anyway would add noise and dilute trust in the clear cases.
- This only works because ADR-002 already has Model A output a full probability
  distribution rather than a single hard label — no architecture change needed to support it.

**Not yet settled:** the exact margin/threshold depends on Model A's probabilities being
reasonably *calibrated* (i.e. "70% confident" actually means right ~70% of the time) — tree
ensembles like HistGB aren't perfectly calibrated out of the box. A calibration check belongs
in `predict.py`/API work, before the exact cutoff number is finalized.

---

## ADR-016: Synthetic Data Regenerated Programmatically with Diagnostic Ambiguity
**Date:** 2026-08-29
**Status:** Accepted

**Decision:**
Replace the LLM-prompt-per-category data generation (`docs/LLM_GENERATION_PROMPT.md`) with a
seeded Python generator, `src/pettriage/data/generate_synthetic.py`, in which every category
carries a probability profile over **all 31 symptoms** rather than its own private symptom
list. Output goes to `data/raw/synthetic_v2/` in the same 23-column schema and same 11
filenames, so notebook 03's validation and merge run unchanged.

**Reason — the first dataset made the problem fake:**
Model A hit 96.4% validation accuracy, which investigation showed was an artifact of the data,
not model quality:
- 79.5% of rows had a symptom combination mapping to exactly **one** category (284 of 310
  distinct combinations). The symptom list *was* the answer — a lookup table.
- Ablation confirmed it: symptoms alone → 90.9%, vitals alone → 26.4%, everything → 96.7%.
- Individual symptoms were already shared (`Weakness` in 10 categories); the giveaway was
  *combinations*, because each category was generated by an isolated prompt naming "its"
  symptoms.
- No row leakage between splits (0 of 330 val rows) — the split was sound; the data wasn't.

**Why that blocked the architecture:** Model A's probabilities were effectively one-hot —
mean top probability 0.984, 89.1% of rows above 0.99, only 0.9% genuinely uncertain. That
makes **ADR-002's soft cascade** meaningless (there is no uncertainty to pass to Model B, only
a hard label in disguise) and **ADR-015's close-alternatives display** untestable (there are
never close alternatives).

**Why a Python generator over rewriting the LLM prompts:**
Reproducible from a seed, ambiguity is a tunable parameter re-runnable in seconds, lives in
git, and is testable in CI — versus 11 manual chatbot runs per iteration that can't be tuned
or reproduced.

**Resulting task difficulty** (measured, `ambiguity=0.0`):
| Metric | Old LLM data | New generator |
|---|---|---|
| Symptom-sets mapping to 1 category | 79.5% | 61.9% |
| Condition top-1 accuracy | 96.4% | 50.9% |
| Condition top-3 accuracy | — | 80.0% |
| Mean top probability | 0.984 | 0.834 |
| Rows above 0.99 confidence | 89.1% | 27.6% |
| Urgency accuracy vs majority baseline | — | 67.0% vs 43.7% |

Top-1 of ~51% is 5.6× random for 11 classes, and the top-3 of 80% is what makes ADR-015
genuinely useful. Urgency emerges from a symptom-severity score rather than being hardcoded
per category, and produces clinically sensible spread on its own (Trauma/Poisoning 68%
Emergency, Eye/Ear 1%, Skin/Fungal 2.5%).

**Note:** the `ambiguity` parameter defaults to `0.0` because the authored profiles already
overlap sufficiently. Raising it blends categories further toward each other and only degrades
the task; it is retained as a knob for studying how models behave as classes become less
separable.

**Side effect — `split_data()` made robust to rare strata.** The new data produces genuinely
rare combinations (a Skin/Fungal case that *is* an emergency), and sklearn refuses to stratify
on a group of 1. `split_data()` now collapses combinations below `min_stratum` (default 6)
to Urgency-only, then to a single shared bucket. Verified to leave the existing dataset's
splits byte-identical.

**Superseded in part by ADR-017**, which raised the dataset to 13,200 rows and added
real-world prevalence imbalance, missing vitals, and label noise.

**The old LLM-generated CSVs are replaced, not kept alongside.** The generator writes to
the same `data/raw/synthetic/` folder using the same 11 filenames as the original
LLM-generated data, overwriting them in place — engineer's call, since the old data's 96.4%
accuracy made it actively misleading to keep around, not just outdated. Nothing downstream
(notebook 03's path, `docs/LLM_GENERATION_PROMPT.md`) needs to change to find the new files;
the manual LLM-prompt workflow itself is superseded by this generator going forward.

---

## ADR-017: LogisticRegression as Model A Baseline; Dataset Scaled Up and Made Messier
**Date:** 2026-08-29
**Status:** Accepted

**Decision:**
1. `LogisticRegression` becomes the **baseline any Model A candidate must beat**, not an
   afterthought. `train_model_a.py` is now a model-selection script: it trains several
   families and picks the champion by validation accuracy rather than assuming boosting wins.
2. The dataset grows from 2,200 to **13,200 rows** and gains three kinds of real-world
   messiness: prevalence-based class imbalance, missing vitals, and label noise.
3. `preprocess.py` gains **median imputation** (fit on train only) to handle the missing
   vitals.

**Reason — the simple model won:**
Measured on the 1,540-row ambiguous dataset:

| Model | Train | Val | Gap | Top-3 |
|---|---|---|---|---|
| HistGB default | 1.000 | 0.524 | 0.476 | 0.809 |
| RandomForest default | 1.000 | 0.548 | 0.452 | 0.833 |
| HistGB regularized | 0.973 | 0.555 | 0.419 | 0.842 |
| **LogisticRegression C=0.3** | **0.616** | **0.591** | **0.025** | **0.870** |

The tree ensembles scored 1.000 on their own training data while getting nearly half the
validation set wrong — they memorised 1,540 specific animals rather than learning disease
patterns. Logistic regression, with almost no train/val gap, generalised better *and* gave
the best top-3. Model capacity should match the structure of the problem, not be maximised.

**Honest caveat:** the generator builds each row from independent per-symptom probabilities,
which gives the data a log-linear structure that logistic regression is naturally suited to.
Real veterinary data may contain genuine feature interactions where trees would win. The
durable lesson is the *method* — always run the simple baseline — not "LogReg is best."

**Reason — dataset size:** a learning curve (same generator, fixed 3,000-row holdout) showed
validation accuracy still climbing well past the old data size, with the train/val gap only
closing once the model could no longer memorise:

| Train rows | Train acc | Val acc | Gap |
|---|---|---|---|
| 1,540 (old) | 1.000 | 0.515 | 0.485 |
| 6,000 | 1.000 | 0.541 | 0.459 |
| 12,000 | 0.754 | 0.583 | 0.171 |
| 19,000 | 0.742 | 0.588 | 0.154 |

53% was therefore *not* the problem's ceiling — a large share of the error was reducible
variance, not irreducible class overlap. Since the data is generated, more of it costs one
command.

**Reason — real-world messiness added:**
- **Prevalence imbalance** (`CATEGORY_PREVALENCE`): real triage queues are not uniform. Skin,
  GI, and ear complaints dominate; systemic viral disease is rare in vaccinated populations.
  Roughly a 5x ratio between the most and least common class. The old uniform 200-per-category
  split quietly taught the model that every disease is equally likely — a false prior that
  would not survive contact with real intake data.
- **Missing vitals** (`--missing-rate`, default 8%): owners frequently cannot report a
  temperature or pulse. Vitals that *are* reported carry measurement error.
- **Label noise** (`--label-noise`, default 3%): a share of recorded diagnoses are initial
  impressions later revised. This places a genuine hard ceiling on achievable accuracy, as
  real labels do.

**Consequence:** `class_weight="balanced"` now matters — without it a model can score well by
neglecting rare classes entirely, so balanced variants are included among the candidates.

**MLflow logging reworked.** Runs are now **top-level, not nested** — the previous parent run
held zero metrics, so clicking it in the UI showed an empty page with no charts, and nested
children did not compare well in the runs table. Each candidate now logs its own params,
headline metrics, per-class F1, and its confusion matrix and classification report as
artifacts. Note the local UI must be started as
`mlflow ui --backend-store-uri sqlite:///mlflow.db` — plain `mlflow ui` reads `./mlruns` and
shows nothing.

---

## ADR-018: Explicit Missing-Value Policy; Macro F1 as the Champion Metric
**Date:** 2026-08-30
**Status:** Accepted

**Decision A — missing values are classified by cause, not blanket-filled.**
`preprocess.py` defines three column lists, and every gap must fall into exactly one:

| List | Meaning | Action |
|---|---|---|
| `IMPUTABLE_NUMERIC_COLS` | "We failed to measure it" — the true value exists, we just don't know it (owner has no thermometer) | Fill with the train-split median |
| `STRUCTURAL_ABSENCE_COLS` | "There is nothing to measure" — an animal with 3 symptoms has no 4th | **Never** fill; multi-hot encoding already represents absence as 0 |
| `REQUIRED_COLS` | "This row is broken" — no `Animal_Type`, no `Disease_Category` | Fail loudly via `validate_required_columns()` |

**Reason:**
The previous implementation imputed every column in `NUMERIC_COLS` indiscriminately. That
happened to be harmless only because the structural columns are text and never reached the
imputer — correctness by accident, not by design. Notebook 03 made the underlying confusion
visible: it reported 785 empty `Symptom_3` and 1,636 empty `Symptom_4` cells as "Missing
values found", then saved anyway and declared the data "cleaned, validated". Those were not
defects; they were animals with fewer symptoms. Filling them would have invented symptoms
that do not exist.

`Age` and `Weight` are classified **imputable rather than required** deliberately: a live
triage API should degrade gracefully when an owner does not know their pet's exact weight,
not reject the request.

`Symptom_Count` is derived (a sum of multi-hot columns) so it can never be missing; it is in
`DERIVED_NUMERIC_COLS`, scaled but never imputed. `NUMERIC_COLS` is now *built* from
`IMPUTABLE_NUMERIC_COLS + DERIVED_NUMERIC_COLS` so the two cannot drift apart.

The lists live in `preprocess.py` and notebook 03 **imports** them rather than redefining
them, so the audit and the pipeline can never disagree about what to do with a gap.

**Decision B — the Model A champion is selected on macro F1, not accuracy.**
`CHAMPION_METRIC = "macro_f1"` in `train_model_a.py`.

**Reason:**
With ~5x class imbalance (ADR-017), accuracy rewards a model for getting the common classes
(Skin/Fungal, Gastrointestinal) right while neglecting rare ones like Viral Systemic
entirely — a model can post a good accuracy number while being useless on exactly the
conditions where a missed diagnosis matters most. Macro F1 averages per-class F1 with equal
weight, so ignoring a rare disease actually costs the model. The two criteria already
disagreed in practice on the first 13,200-row run (one candidate led on accuracy while
another led on macro F1), so the choice is not academic. `train_model_a.py` prints an
explicit note whenever they disagree, naming both, rather than hiding the tension.

**Not done (deliberately):** missingness-indicator columns — adding a binary
`Body_Temperature_Missing` feature so the model can learn from the *fact* that a value was
absent. This is standard practice and may well help (an owner who cannot measure a
temperature may differ systematically from one who can), but it is a separate modelling
decision and was left out rather than smuggled in alongside the policy change.

---

## ADR-019: Cardinal Symptoms Made Prominent; Label Noise Removed; Dataset Scaled to 30k
**Date:** 2026-08-30
**Status:** Accepted

**Decision:**
1. Rewrite `SYMPTOM_PROFILES` so each category's **cardinal signs** (the findings that
   define its diseases) sit at 0.40-0.65, while nonspecific malaise symptoms (Lethargy,
   Loss of Appetite, Fever, Weakness) drop to 0.10-0.40.
2. `--label-noise` now defaults to **0.0**.
3. `--total-rows` raised to **30,000**.

**Reason 1 — the profiles were built backwards.**
The original design gave the highest weights to the symptoms every category shares, and left
the distinguishing signs at 0.15-0.25. Since each row draws only 2-4 symptoms, most rows came
out as generic malaise and the discriminating sign often was not drawn at all. Viral Systemic
and Bacterial/Parasitic reached **0.914** cosine similarity — the highest of all 55 pairs —
and Viral Systemic's F1 collapsed to 0.209.

That was not clinical reality. Lyme disease's limping and joint swelling, FIP's abdominal
effusion, and distemper's nasal/ocular discharge are *defining presentations*, not incidental
extras. The fix raises them accordingly and trims the shared malaise weights.

**Reason 2 — label noise was pure handicap.** Simulating wrong labels only pays off if
something in the pipeline handles them (robust losses, label smoothing, confident learning);
nothing here does. Measured cost was ~0.028 macro F1, and it made every failure ambiguous
between "our model" and "noise we injected ourselves". It also lands hardest exactly where
the model was weakest — rare classes absorb noise from ten other categories with few true
rows to dilute it (Viral Systemic 9.1% wrong labels vs Skin/Fungal 1.7%). Real data will
bring its own noise; the parameter is retained for deliberate robustness experiments.
Missing vitals stay ON, because they exercise real machinery (the imputer).

**Reason 3 — class size was the strongest predictor of per-class F1** (r = 0.77). At 30,000
rows the rarest class gets ~630 training rows instead of 303.

**Measured effect** (LogisticRegression `class_weight='balanced'`, validation split):

| Class | Train rows | Old F1 | New F1 | Change |
|---|---|---|---|---|
| Viral Systemic | 630 | 0.209 | 0.367 | **+0.158** |
| Bacterial / Parasitic | 1,680 | 0.393 | 0.542 | **+0.149** |
| Renal / Urinary | 1,470 | 0.492 | 0.621 | +0.129 |
| Eye / Ear | 2,519 | 0.664 | 0.759 | +0.095 |
| Gastrointestinal | 3,150 | 0.581 | 0.659 | +0.078 |
| Trauma / Poisoning | 1,260 | 0.500 | 0.576 | +0.076 |
| Dental / Oral | 1,890 | 0.679 | 0.748 | +0.069 |
| Skin / Fungal | 3,150 | 0.749 | 0.796 | +0.047 |
| Musculoskeletal | 2,310 | 0.660 | 0.702 | +0.042 |
| Endocrine / Metabolic | 841 | 0.457 | 0.491 | +0.034 |
| Respiratory | 2,100 | 0.712 | 0.735 | +0.023 |

Every class improved. **Macro F1 0.554 -> 0.636**, accuracy 0.598 -> 0.672, top-3 0.860 ->
0.902. Viral <-> Bacterial similarity fell from 0.914 to **0.639**.

**The problem stayed genuinely hard, which was a requirement, not an accident.** Only 33.2%
of symptom combinations map to a single category (the discredited LLM dataset was 79.5%),
accuracy is 0.672 rather than the fake 0.964, and 44.3% of predictions remain genuinely
uncertain (top probability below 0.6) — so ADR-002's soft cascade and ADR-015's
close-alternatives display still have real uncertainty to work with.

**Still weak, and knowingly so:** Viral Systemic (0.367) and Endocrine / Metabolic (0.491)
remain the two worst classes. Endocrine overlaps Renal / Urinary at 0.826 cosine — the
highest remaining pair — but that overlap is clinically legitimate (diabetes and kidney
disease both cause polyuria and polydipsia), separated mainly by Difficulty Urinating. These
are candidates for the hyperparameter tuning that follows, not for further data surgery.

**Sequencing note:** hyperparameter tuning was deliberately deferred until after this change.
Hyperparameters are tuned *for a dataset*; retuning would have been necessary anyway once the
distribution moved, so tuning first would have meant doing the work twice.

---

## ADR-020: Two-Stage Prediction — Model Outputs Probabilities, a Separate Rule Decides
**Date:** 2026-08-30
**Status:** Accepted in principle — implement in `predict.py` / the API layer (not built yet)

**Decision:**
Do not bake the precision/recall tradeoff into model training. Split prediction in two:

- **Stage 1 — the model.** Outputs a calibrated probability across all 11 categories.
  Trained once, optimised for good probabilities (macro F1 as the selection metric).
- **Stage 2 — the decision rule.** Converts those probabilities into what the user is
  shown. This is where the precision/recall tradeoff is set, and it can be changed
  **without retraining anything.**

**Reason:**
Tuning `class_weight` to rescue rare classes has a real cost, measured on Viral Systemic
(135 validation cases):

| Config | Caught | Missed | False alarms | Precision | Recall |
|---|---|---|---|---|---|
| No `class_weight` | 34 | 101 | 47 | 0.420 | 0.252 |
| `class_weight='balanced'` | 67 | 68 | **169** | 0.284 | 0.496 |

Doubling the catches tripled the false alarms, and *lowered* macro F1 (0.645 -> 0.641).
Multiclass predictions are zero-sum: 122 extra rows pulled into "Viral" are 122 rows stolen
from the classes they actually belong to, costing those classes recall.

The two-stage split avoids the dilemma entirely. **ADR-015's "show close alternatives" is
already a stage-2 rule**, and it recovers the rare-class recall for free:

| Class | top-1 recall | top-3 recall |
|---|---|---|
| Viral Systemic | 0.252 | **0.711** |
| Endocrine / Metabolic | 0.436 | 0.777 |
| Bacterial / Parasitic | 0.494 | 0.892 |

Viral Systemic goes from catching 25% to 71% of real cases with no retraining, no
`class_weight`, and no false positives stolen from other categories. The information was in
the probability distribution; top-1 accuracy simply discarded it.

**Consequences:**
- `class_weight` stops being a metric argument and becomes an ordinary hyperparameter for the
  search to settle empirically.
- Macro F1 stays the model-selection metric (ADR-018 stands); per-class recall is logged for
  visibility rather than optimised directly.
- Calibration matters more than previously assumed — a stage-2 rule keyed on probability
  thresholds is only sound if the probabilities mean what they say. Calibration check belongs
  with `predict.py`.

**Real-world precedent:** mammography AI (model scores, policy sets the recall threshold,
radiologist reviews flagged cases), fraud detection (threshold set by review-team capacity),
and the cautionary case of hospital sepsis alerts, where thresholds set too aggressively
produced alert fatigue and clinicians began ignoring them.

---

## ADR-021: Hyperparameter Tuning Plan (Deferred — Next Session)
**Date:** 2026-08-30
**Status:** Agreed, not yet implemented

**Decision:** Implement hyperparameter search for Model A as the next modelling step, with:

1. **`RandomizedSearchCV` with cross-validation on the TRAIN split**, scored on `f1_macro`.
   Not tuning against the validation set: trying many configurations and keeping whichever
   scores best on val fits the hyperparameters *to val*, and val then stops being an honest
   estimate. Val is for the final champion comparison; test stays sealed.
2. **Fix the `class_weight` gap.** `HistGradientBoostingClassifier` supports `class_weight`
   (sklearn 1.9) and the current candidate does not set it, while both LogisticRegression
   variants and RandomForest do. It won the last run without the tool its competitors had.
   Add it to the search space rather than hardcoding it.
3. **Log every search's best params and score to MLflow**, plus per-class recall.

**Sequencing reason:** tuning was deliberately deferred until after the dataset fix
(ADR-019). Hyperparameters are tuned *for a dataset*; moving the distribution would have
forced a full retune, so tuning first would have meant doing the work twice.

**Realistic expectation:** roughly +0.03-0.05 macro F1. Note the current four candidates
finished within 0.011 of each other (0.636-0.647) — that is a statistical tie, not a decisive
champion, so treat the current "HistGB wins" result as provisional.

---

## ADR-022: Model B Cascade — Probability Extraction Still Undecided
**Date:** 2026-08-30
**Status:** Open question — must be settled before Model B is built

**Decision required:** which rows' Model A probabilities become Model B's input features.

**Option A — in-sample.** Call `model_a.predict_proba()` on the same training rows Model A
was fit on. Simple, but a model is always more confident about data it trained on, so Model B
would learn from unrealistically sharp distributions and may over-trust them in production.

**Option B — cross-fitted (out-of-fold).** Split train into folds, train a temporary Model A
on each fold-minus-one, predict the held-out fold, stitch the results together. Model B then
sees probabilities of the same quality it will meet at inference time. More moving parts.

**Now viable, which it previously was not.** On the discredited dataset Model A's output was
effectively one-hot (mean top probability 0.984, 89% of rows above 0.99), which made ADR-002's
soft cascade meaningless. After ADR-019 the champion's mean top probability is ~0.66 with
44.3% of predictions genuinely uncertain (below 0.6) — real uncertainty for Model B to
consume.

**Deliberately not decided unilaterally** — this is the "how does Model A's probability get
extracted for B" question the engineer asked to walk through separately.
