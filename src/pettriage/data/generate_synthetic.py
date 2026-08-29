"""
Generates the synthetic training dataset with realistic diagnostic ambiguity.

Replaces the original 11-separate-LLM-prompts approach (docs/LLM_GENERATION_PROMPT.md),
which produced data where 79.5% of symptom combinations mapped to exactly one disease
category — a lookup table, not a diagnosis problem. Model A scored 96.4% on it, and its
predicted probabilities were effectively one-hot (89% of rows above 0.99 confidence),
which made the ADR-002 soft cascade and ADR-015 alternative-suggestions impossible to
evaluate. See ADR-016.

The fix: every category gets a probability profile over ALL 31 symptoms rather than its
own private symptom list. Profiles overlap heavily by design, so "Vomiting + Lethargy"
genuinely occurs under Gastrointestinal, Renal/Urinary, Viral Systemic and
Trauma/Poisoning. Partial discrimination comes from vitals, age skew, and duration —
never from the symptom set alone.

Run: python -m pettriage.data.generate_synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pettriage.config import RANDOM_STATE, RAW_DATA_DIR

SYMPTOMS = [
    "Fever", "Lethargy", "Loss of Appetite", "Coughing", "Nasal Discharge",
    "Vomiting", "Diarrhea", "Sneezing", "Weight Loss", "Labored Breathing",
    "Swollen Joints", "Eye Discharge", "Skin Lesions", "Pale Gums", "Bloody Stool",
    "Weakness", "Seizures", "Excessive Thirst", "Jaundice", "Limping",
    "Excessive Scratching", "Hair Loss", "Bad Breath", "Drooling", "Bleeding",
    "Difficulty Urinating", "Increased Urination", "Swollen Abdomen", "Head Shaking",
    "Ear Scratching", "Collapse",
]

# Per-category probability of each symptom appearing.
#
# Design rule (revised — see ADR-019): each category's CARDINAL signs — the findings
# that actually define its diseases — sit at 0.40-0.65, while the nonspecific malaise
# symptoms (Lethargy, Loss of Appetite, Fever, Weakness) sit lower at 0.10-0.40.
#
# The first version had this backwards: the shared generic symptoms carried the highest
# weights and the distinguishing ones sat at 0.15-0.25. Since each row draws only 2-4
# symptoms, most rows ended up as generic malaise and the discriminating sign frequently
# was not drawn at all — Viral Systemic and Bacterial/Parasitic reached 0.914 cosine
# similarity and F1 collapsed to 0.21. That was not clinical reality: Lyme disease's
# limping and joint swelling, FIP's swollen abdomen, and distemper's nasal/ocular
# discharge are defining presentations, not incidental extras.
#
# Genuine ambiguity is preserved deliberately — a fever-and-lethargy-only case should
# still be hard to place, because it is hard in real life.
SYMPTOM_PROFILES: dict[str, dict[str, float]] = {
    # Kennel cough, feline URI, asthma, bronchitis: airway signs dominate.
    "Respiratory": {
        "Coughing": 0.60, "Nasal Discharge": 0.50, "Sneezing": 0.45,
        "Labored Breathing": 0.40, "Eye Discharge": 0.20, "Lethargy": 0.30,
        "Loss of Appetite": 0.25, "Fever": 0.25, "Weakness": 0.12,
        "Weight Loss": 0.08, "Pale Gums": 0.05, "Collapse": 0.03,
        "Vomiting": 0.06, "Drooling": 0.05,
    },
    # Parvo, gastroenteritis, pancreatitis, IBD: the gut signs are the presentation.
    "Gastrointestinal": {
        "Vomiting": 0.65, "Diarrhea": 0.60, "Bloody Stool": 0.25,
        "Loss of Appetite": 0.45, "Lethargy": 0.40, "Swollen Abdomen": 0.20,
        "Drooling": 0.18, "Fever": 0.22, "Weakness": 0.25, "Weight Loss": 0.18,
        "Pale Gums": 0.12, "Collapse": 0.06, "Jaundice": 0.05,
        "Excessive Thirst": 0.06,
    },
    # Distemper (nasal/ocular discharge + neurological), FIP (abdominal effusion),
    # FeLV/FIV (chronic wasting, anaemia). These signs are characteristic, not extras.
    "Viral Systemic": {
        "Nasal Discharge": 0.40, "Eye Discharge": 0.40, "Weight Loss": 0.45,
        "Swollen Abdomen": 0.35, "Seizures": 0.25, "Pale Gums": 0.18,
        "Fever": 0.40, "Lethargy": 0.45, "Loss of Appetite": 0.35,
        "Weakness": 0.28, "Vomiting": 0.18, "Diarrhea": 0.18, "Coughing": 0.12,
        "Labored Breathing": 0.12, "Collapse": 0.08, "Jaundice": 0.08,
    },
    # Lyme (limping, joint swelling), leptospirosis (jaundice, renal), heartworm
    # (cough), intestinal parasites (diarrhoea, wasting).
    "Bacterial / Parasitic": {
        "Limping": 0.45, "Swollen Joints": 0.40, "Jaundice": 0.30,
        "Excessive Thirst": 0.28, "Coughing": 0.30, "Diarrhea": 0.35,
        "Weight Loss": 0.35, "Fever": 0.35, "Lethargy": 0.35,
        "Loss of Appetite": 0.30, "Weakness": 0.25, "Vomiting": 0.20,
        "Labored Breathing": 0.15, "Pale Gums": 0.12, "Bloody Stool": 0.12,
    },
    # Ringworm, dermatitis, mange: dermatological signs, few systemic ones.
    "Skin / Fungal": {
        "Excessive Scratching": 0.65, "Skin Lesions": 0.60, "Hair Loss": 0.50,
        "Bleeding": 0.10, "Ear Scratching": 0.15, "Head Shaking": 0.10,
        "Lethargy": 0.15, "Loss of Appetite": 0.10, "Fever": 0.08,
        "Weight Loss": 0.05, "Weakness": 0.06,
    },
    # Arthritis, dysplasia, fracture, IVDD: locomotor signs without systemic illness —
    # which is what separates this from Lyme (limping WITH fever/jaundice).
    "Musculoskeletal": {
        "Limping": 0.65, "Swollen Joints": 0.40, "Weakness": 0.40,
        "Lethargy": 0.28, "Loss of Appetite": 0.20, "Collapse": 0.12,
        "Fever": 0.10, "Bleeding": 0.08, "Weight Loss": 0.10, "Drooling": 0.05,
        "Pale Gums": 0.05,
    },
    # Otitis, ear mites, conjunctivitis, glaucoma: localised to eye/ear.
    "Eye / Ear": {
        "Ear Scratching": 0.55, "Head Shaking": 0.50, "Eye Discharge": 0.50,
        "Excessive Scratching": 0.15, "Skin Lesions": 0.10, "Hair Loss": 0.06,
        "Nasal Discharge": 0.10, "Lethargy": 0.18, "Loss of Appetite": 0.12,
        "Fever": 0.10, "Drooling": 0.04,
    },
    # UTI, blockage, CKD, stones. Shares polyuria/polydipsia with Endocrine — that is
    # clinically real — but Difficulty Urinating is near-exclusive to this category.
    "Renal / Urinary": {
        "Difficulty Urinating": 0.55, "Increased Urination": 0.50,
        "Excessive Thirst": 0.50, "Vomiting": 0.30, "Weight Loss": 0.28,
        "Bad Breath": 0.15, "Swollen Abdomen": 0.15, "Lethargy": 0.35,
        "Loss of Appetite": 0.32, "Weakness": 0.25, "Pale Gums": 0.10,
        "Fever": 0.15, "Collapse": 0.08, "Bloody Stool": 0.05,
    },
    # Diabetes, thyroid, Cushing's: polyuria/polydipsia plus weight change and coat
    # problems, but NOT dysuria — that is the Renal discriminator.
    "Endocrine / Metabolic": {
        "Excessive Thirst": 0.60, "Increased Urination": 0.55, "Weight Loss": 0.50,
        "Hair Loss": 0.22, "Lethargy": 0.32, "Loss of Appetite": 0.22,
        "Weakness": 0.30, "Vomiting": 0.15, "Skin Lesions": 0.12,
        "Swollen Abdomen": 0.12, "Collapse": 0.08, "Seizures": 0.06, "Fever": 0.08,
    },
    # Periodontal disease, stomatitis, abscess, oral tumour.
    "Dental / Oral": {
        "Bad Breath": 0.65, "Drooling": 0.55, "Loss of Appetite": 0.50,
        "Bleeding": 0.25, "Weight Loss": 0.25, "Lethargy": 0.22, "Fever": 0.12,
        "Pale Gums": 0.06, "Vomiting": 0.06, "Swollen Joints": 0.03,
    },
    # HBC, toxin ingestion, heatstroke, bites: acute catastrophic signs.
    "Trauma / Poisoning": {
        "Collapse": 0.45, "Pale Gums": 0.35, "Bleeding": 0.35, "Seizures": 0.30,
        "Vomiting": 0.35, "Labored Breathing": 0.30, "Limping": 0.25,
        "Weakness": 0.35, "Lethargy": 0.30, "Loss of Appetite": 0.22,
        "Drooling": 0.20, "Bloody Stool": 0.15, "Skin Lesions": 0.15,
        "Diarrhea": 0.12, "Fever": 0.10,
    },
}

# Age skew per category: (mean, std) in years, clipped to 1-15. Older animals skew
# toward chronic disease, younger toward infectious — a weak but real signal.
AGE_PROFILES = {
    "Respiratory": (5.0, 3.5), "Gastrointestinal": (5.0, 4.0),
    "Viral Systemic": (3.5, 3.0), "Bacterial / Parasitic": (4.5, 3.5),
    "Skin / Fungal": (5.5, 3.5), "Musculoskeletal": (8.5, 4.0),
    "Eye / Ear": (6.0, 4.0), "Renal / Urinary": (9.5, 3.5),
    "Endocrine / Metabolic": (9.5, 3.0), "Dental / Oral": (9.0, 3.5),
    "Trauma / Poisoning": (5.0, 4.0),
}

# Weights over (acute, subacute, chronic) duration pools per category.
DURATION_PROFILES = {
    "Respiratory": (0.20, 0.55, 0.25), "Gastrointestinal": (0.45, 0.40, 0.15),
    "Viral Systemic": (0.20, 0.45, 0.35), "Bacterial / Parasitic": (0.15, 0.45, 0.40),
    "Skin / Fungal": (0.10, 0.35, 0.55), "Musculoskeletal": (0.30, 0.30, 0.40),
    "Eye / Ear": (0.20, 0.45, 0.35), "Renal / Urinary": (0.25, 0.35, 0.40),
    "Endocrine / Metabolic": (0.05, 0.25, 0.70), "Dental / Oral": (0.05, 0.25, 0.70),
    "Trauma / Poisoning": (0.75, 0.20, 0.05),
}

DURATION_POOLS = {
    "acute": ["A few hours", "Today", "1 day", "2 days", "3 days"],
    "subacute": ["4 days", "5 days", "1 week", "10 days", "2 weeks"],
    "chronic": [
        "3 weeks", "1 month", "6 weeks", "2 months", "3 months", "6 months", "1 year",
    ],
}

# Realistic relative prevalence — real triage queues are NOT balanced. Skin, GI and
# ear complaints dominate a clinic's day; systemic viral disease is comparatively rare
# in vaccinated populations. Deliberately different from the old 200-per-category
# uniform split, which quietly taught the model that every disease is equally likely.
CATEGORY_PREVALENCE = {
    "Skin / Fungal": 0.15,
    "Gastrointestinal": 0.15,
    "Eye / Ear": 0.12,
    "Musculoskeletal": 0.11,
    "Respiratory": 0.10,
    "Dental / Oral": 0.09,
    "Bacterial / Parasitic": 0.08,
    "Renal / Urinary": 0.07,
    "Trauma / Poisoning": 0.06,
    "Endocrine / Metabolic": 0.04,
    "Viral Systemic": 0.03,
}

# Share of cats vs dogs per category.
CAT_SHARE = {
    "Respiratory": 0.60, "Gastrointestinal": 0.40, "Viral Systemic": 0.55,
    "Bacterial / Parasitic": 0.30, "Skin / Fungal": 0.50, "Musculoskeletal": 0.40,
    "Eye / Ear": 0.50, "Renal / Urinary": 0.55, "Endocrine / Metabolic": 0.50,
    "Dental / Oral": 0.50, "Trauma / Poisoning": 0.45,
}

DOG_BREEDS = {
    "Labrador": 32, "German Shepherd": 34, "Golden Retriever": 31, "Bulldog": 24,
    "Poodle": 20, "Beagle": 12, "Rottweiler": 50, "Dachshund": 8, "Chihuahua": 3,
    "Shih Tzu": 6, "Husky": 24, "Border Collie": 18, "Doberman": 38, "Boxer": 30,
    "Mixed Breed": 20,
}
CAT_BREEDS = {
    "Siamese": 4.5, "Persian": 4.5, "Maine Coon": 7.0, "British Shorthair": 5.5,
    "Bengal": 5.0, "Ragdoll": 6.5, "Domestic Shorthair": 4.5,
    "Domestic Longhair": 5.0, "Sphynx": 3.8, "Burmese": 4.2,
}

# How much each symptom contributes to the urgency score.
SEVERITY_WEIGHTS = {
    "Collapse": 3.0, "Seizures": 3.0, "Pale Gums": 2.5, "Labored Breathing": 2.0,
    "Bloody Stool": 1.8, "Jaundice": 1.5, "Bleeding": 1.5,
    "Difficulty Urinating": 1.4, "Weakness": 0.7, "Vomiting": 0.5, "Diarrhea": 0.5,
    "Fever": 0.5, "Swollen Abdomen": 0.6, "Lethargy": 0.3, "Loss of Appetite": 0.3,
    "Drooling": 0.2, "Weight Loss": 0.2, "Limping": 0.2, "Coughing": 0.2,
}

# Baseline risk each category carries before symptoms are considered.
CATEGORY_BASE_RISK = {
    "Trauma / Poisoning": 2.0, "Viral Systemic": 1.2, "Renal / Urinary": 1.0,
    "Gastrointestinal": 0.8, "Bacterial / Parasitic": 0.8, "Respiratory": 0.6,
    "Endocrine / Metabolic": 0.5, "Musculoskeletal": 0.5, "Dental / Oral": 0.3,
    "Eye / Ear": 0.2, "Skin / Fungal": 0.1,
}

# Symptom text -> the legacy Yes/No flag column it must agree with. Kept so the
# generated schema still matches what notebook 03's consistency checks expect.
SYMPTOM_TO_FLAG = {
    "Loss of Appetite": "Appetite_Loss", "Vomiting": "Vomiting", "Diarrhea": "Diarrhea",
    "Coughing": "Coughing", "Labored Breathing": "Labored_Breathing",
    "Limping": "Lameness", "Skin Lesions": "Skin_Lesions",
    "Excessive Scratching": "Skin_Lesions", "Nasal Discharge": "Nasal_Discharge",
    "Eye Discharge": "Eye_Discharge",
}
FLAG_COLS = [
    "Appetite_Loss", "Vomiting", "Diarrhea", "Coughing", "Labored_Breathing",
    "Lameness", "Skin_Lesions", "Nasal_Discharge", "Eye_Discharge",
]

COLUMNS = [
    "Animal_Type", "Breed", "Age", "Gender", "Weight", "Symptom_1", "Symptom_2",
    "Symptom_3", "Symptom_4", "Duration", *FLAG_COLS, "Body_Temperature",
    "Heart_Rate", "Disease_Category", "Urgency",
]

CATEGORY_FILENAMES = {
    "Respiratory": "respiratory.csv",
    "Gastrointestinal": "gastrointestinal.csv",
    "Viral Systemic": "viral_systemic.csv",
    "Bacterial / Parasitic": "bacterial_parasitic.csv",
    "Skin / Fungal": "skin_fungal.csv",
    "Musculoskeletal": "musculoskeletal.csv",
    "Eye / Ear": "eye_ear.csv",
    "Renal / Urinary": "renal_urinary.csv",
    "Endocrine / Metabolic": "endocrine_metabolic.csv",
    "Dental / Oral": "dental.csv",
    "Trauma / Poisoning": "trauma_poisoning.csv",
}


# Turns a category's sparse profile into a full 31-length weight vector, then blends it
# toward the average of all categories. `ambiguity` is the difficulty knob: 0.0 keeps
# each category's own profile, 1.0 makes every category identical (and the task
# impossible). Blending is what creates overlapping presentations.
def _blended_weights(category: str, ambiguity: float) -> np.ndarray:
    def vec(cat):
        p = SYMPTOM_PROFILES[cat]
        return np.array([p.get(s, 0.02) for s in SYMPTOMS], dtype=float)

    own = vec(category)
    average = np.mean([vec(c) for c in SYMPTOM_PROFILES], axis=0)
    blended = (1.0 - ambiguity) * own + ambiguity * average
    return blended / blended.sum()


# Draws 2-4 distinct symptoms for one row, weighted by the blended profile. Sampling
# without replacement from overlapping weights is what lets the same combination show
# up under different categories.
def _draw_symptoms(rng: np.random.Generator, weights: np.ndarray) -> list[str]:
    n = rng.choice([2, 3, 4], p=[0.35, 0.40, 0.25])
    idx = rng.choice(len(SYMPTOMS), size=n, replace=False, p=weights)
    return [SYMPTOMS[i] for i in idx]


# Body temperature and heart rate. Both overlap heavily between healthy and sick
# animals; fever lifts temperature only when a fever-ish symptom is actually present,
# so vitals stay a partial signal rather than another giveaway.
def _draw_vitals(
    rng: np.random.Generator, is_cat: bool, symptoms: list[str], severity: float
) -> tuple[float, int]:
    base_temp = 38.6 if is_cat else 38.7
    temp = rng.normal(base_temp, 0.35)
    if "Fever" in symptoms:
        temp += rng.uniform(0.8, 2.2)
    if severity > 4.0:
        temp += rng.uniform(-0.3, 1.2)
    temp = float(np.clip(temp, 36.5, 42.0))

    base_hr = rng.normal(180, 28) if is_cat else rng.normal(100, 22)
    hr = base_hr + severity * rng.uniform(3, 9)
    hr = int(np.clip(hr, 55, 265))
    return round(temp, 1), hr


# Converts the severity score into a triage label. The thresholds carry deliberate
# noise so urgency is strongly but not perfectly determined by the score — a real
# borderline case can plausibly land either side.
def _assign_urgency(rng: np.random.Generator, severity: float) -> str:
    score = severity + rng.normal(0, 0.9)
    if score >= 4.2:
        return "Emergency"
    if score >= 1.9:
        return "Monitor"
    return "Okay"


# Builds one category's worth of rows.
#
# `missing_rate` and `label_noise` inject the messiness real intake data always has:
#   - Owners often can't report a temperature or pulse (no thermometer at home), so
#     those vitals go missing rather than being silently invented.
#   - Vitals that ARE reported carry measurement error — an owner counting a pulse is
#     less accurate than a clinic monitor.
#   - A small share of recorded diagnoses are simply wrong (initial impression later
#     revised). This puts a hard ceiling on achievable accuracy, exactly as real
#     labels do.
def generate_category(
    category: str,
    n_rows: int,
    ambiguity: float,
    rng: np.random.Generator,
    missing_rate: float = 0.0,
    label_noise: float = 0.0,
    other_categories: list[str] | None = None,
) -> pd.DataFrame:
    weights = _blended_weights(category, ambiguity)
    age_mean, age_std = AGE_PROFILES[category]
    dur_weights = DURATION_PROFILES[category]
    rows = []

    for _ in range(n_rows):
        is_cat = rng.random() < CAT_SHARE[category]
        breeds = CAT_BREEDS if is_cat else DOG_BREEDS
        breed = str(rng.choice(list(breeds)))
        weight = round(max(1.5, rng.normal(breeds[breed], breeds[breed] * 0.18)), 1)
        age = int(np.clip(round(rng.normal(age_mean, age_std)), 1, 15))

        symptoms = _draw_symptoms(rng, weights)
        severity = CATEGORY_BASE_RISK[category] + sum(
            SEVERITY_WEIGHTS.get(s, 0.15) for s in symptoms
        )
        temp, hr = _draw_vitals(rng, is_cat, symptoms, severity)
        if temp > 40.5:
            severity += 1.5
        urgency = _assign_urgency(rng, severity)

        pool = str(rng.choice(["acute", "subacute", "chronic"], p=dur_weights))
        duration = str(rng.choice(DURATION_POOLS[pool]))

        padded = symptoms + [""] * (4 - len(symptoms))
        flags = {c: "No" for c in FLAG_COLS}
        for s in symptoms:
            if s in SYMPTOM_TO_FLAG:
                flags[SYMPTOM_TO_FLAG[s]] = "Yes"

        # Owner-reported vitals are noisy, and often simply absent.
        temp_out: float | str = temp
        hr_out: int | str = hr
        if missing_rate:
            if rng.random() < missing_rate:
                temp_out = ""
            else:
                temp_out = round(float(temp + rng.normal(0, 0.15)), 1)
            if rng.random() < missing_rate:
                hr_out = ""
            else:
                hr_out = int(np.clip(hr + rng.normal(0, 6), 40, 280))

        # A small share of records carry the wrong diagnosis.
        label = category
        if label_noise and other_categories and rng.random() < label_noise:
            label = str(rng.choice(other_categories))

        rows.append({
            "Animal_Type": "Cat" if is_cat else "Dog",
            "Breed": breed,
            "Age": age,
            "Gender": str(rng.choice(["Male", "Female"])),
            "Weight": weight,
            "Symptom_1": padded[0], "Symptom_2": padded[1],
            "Symptom_3": padded[2], "Symptom_4": padded[3],
            "Duration": duration,
            **flags,
            "Body_Temperature": temp_out,
            "Heart_Rate": hr_out,
            "Disease_Category": label,
            "Urgency": urgency,
        })

    return pd.DataFrame(rows, columns=COLUMNS)


# Splits `total_rows` across categories by real-world prevalence (or evenly when
# `balanced` is set), then writes one CSV per category using the same filenames the
# original LLM workflow used, so notebook 03 merges them unchanged.
def generate_all(
    out_dir: Path,
    total_rows: int,
    ambiguity: float,
    seed: int,
    balanced: bool = False,
    missing_rate: float = 0.0,
    label_noise: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = list(CATEGORY_FILENAMES)

    parts = []
    for category, filename in CATEGORY_FILENAMES.items():
        share = 1 / len(categories) if balanced else CATEGORY_PREVALENCE[category]
        n_rows = max(30, round(total_rows * share))
        part = generate_category(
            category, n_rows, ambiguity, rng,
            missing_rate=missing_rate,
            label_noise=label_noise,
            other_categories=[c for c in categories if c != category],
        )
        part.to_csv(out_dir / filename, index=False)
        parts.append(part)
        print(f"  {category:24s} -> {filename} ({len(part)} rows)")

    combined = pd.concat(parts, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows across {len(categories)} categories")
    mix = combined["Urgency"].value_counts(normalize=True).round(3).to_dict()
    print("Urgency mix:", mix)
    if missing_rate:
        miss_t = (combined["Body_Temperature"] == "").mean()
        miss_h = (combined["Heart_Rate"] == "").mean()
        print(f"Missing vitals: Body_Temperature {miss_t:.1%}, Heart_Rate {miss_h:.1%}")
    if label_noise:
        print(f"Label noise applied at {label_noise:.1%}")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Generate ambiguous synthetic pet triage data"
    )
    # Same folder + same 11 filenames as the original LLM-generated data (ADR-007) —
    # running this overwrites those files in place, so notebook 03 and everything
    # downstream needs no path changes.
    parser.add_argument("--out-dir", type=Path, default=RAW_DATA_DIR / "synthetic")
    # Sized so the RAREST class (Viral Systemic, 3% prevalence) gets ~900 rows total /
    # ~630 in train, roughly double what 13,200 gave it. Rare-class F1 tracked class
    # size at r=0.77, so this is the direct lever on the weakest categories.
    parser.add_argument("--total-rows", type=int, default=30000)
    # Split rows by real-world prevalence by default; --balanced forces the old
    # equal-rows-per-category behaviour for comparison.
    parser.add_argument("--balanced", action="store_true")
    # 0.0 keeps the clinical profiles as authored — they already overlap enough to give
    # ~51% top-1 / ~80% top-3. Raising it blends categories further toward each other,
    # which only makes the task harder without adding useful signal; kept as a knob for
    # experimenting with how model behaviour degrades as classes become less separable.
    parser.add_argument("--ambiguity", type=float, default=0.0)
    # Share of rows where an owner-reported vital is simply absent. Kept ON: missing
    # vitals exercise real machinery (the imputer in preprocess.py).
    parser.add_argument("--missing-rate", type=float, default=0.08)
    # Share of rows whose recorded diagnosis is wrong. Defaults to 0 (ADR-019).
    # Simulating label noise only pays off if something in the pipeline handles it —
    # nothing here does, so it was pure handicap: it cost ~0.028 macro F1 and made
    # failures ambiguous between "our model" and "noise we poured in ourselves". It
    # lands hardest on rare classes too (9.1% wrong labels for Viral Systemic vs 1.7%
    # for Skin/Fungal), since they absorb noise from ten other classes with few true
    # rows to dilute it. Kept as a knob for deliberate robustness experiments.
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    print(f"Generating to {args.out_dir}")
    print(f"  total_rows={args.total_rows}, balanced={args.balanced}, "
          f"ambiguity={args.ambiguity}")
    print(f"  missing_rate={args.missing_rate}, label_noise={args.label_noise}, "
          f"seed={args.seed}")
    generate_all(
        args.out_dir,
        args.total_rows,
        args.ambiguity,
        args.seed,
        balanced=args.balanced,
        missing_rate=args.missing_rate,
        label_noise=args.label_noise,
    )


if __name__ == "__main__":
    main()
