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
Symptoms: vomiting + pale gums + lethargy
Hard label approach → Model A: "Stomach Upset" → Model B: "Monitor" ← WRONG, dangerous

Soft label approach → Model A: {Internal Bleeding: 0.35, Stomach Upset: 0.30, ...}
                   → Model B sees uncertainty → more likely: "Emergency" ← correct
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
