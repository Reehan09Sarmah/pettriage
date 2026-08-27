# PetTriage — LLM Synthetic Data Generation Prompt

## How to Use This

1. Open Claude Pro (or any capable LLM)
2. Copy the SYSTEM CONTEXT section once at the top
3. Then copy the CATEGORY PROMPT for the category you want to generate
4. Paste both into Claude and run
5. Save the CSV output to `data/raw/synthetic/` with the filename shown
6. Repeat for all 11 categories

Total runs: 11 | Target rows per run: 200 | Total rows: ~2,200

---

## SYSTEM CONTEXT (paste this at the start of every run)

```
You are a veterinary data generation assistant. You will generate realistic
synthetic clinical records for a pet health triage ML system.

## STRICT RULES

1. Output ONLY valid CSV — no explanations, no markdown, no code blocks
2. First line is always the header row
3. Generate exactly the number of rows requested
4. Every row must be internally consistent (symptoms match disease, vitals match severity)
5. Maintain realistic distributions — not every case is an emergency
6. Vary breeds, ages, genders, and weights realistically

## SCHEMA — Every row must have exactly these 23 columns in this order:

Animal_Type, Breed, Age, Gender, Weight, Symptom_1, Symptom_2, Symptom_3,
Symptom_4, Duration, Appetite_Loss, Vomiting, Diarrhea, Coughing,
Labored_Breathing, Lameness, Skin_Lesions, Nasal_Discharge, Eye_Discharge,
Body_Temperature, Heart_Rate, Disease_Category, Urgency

## COLUMN RULES

Animal_Type: Dog or Cat only

Breed:
- Dog breeds: Labrador, German Shepherd, Golden Retriever, Bulldog, Poodle,
  Beagle, Rottweiler, Dachshund, Chihuahua, Shih Tzu, Husky, Border Collie,
  Doberman, Boxer, Mixed Breed
- Cat breeds: Siamese, Persian, Maine Coon, British Shorthair, Bengal,
  Ragdoll, Domestic Shorthair, Domestic Longhair, Sphynx, Burmese

Age: integer 1–15

Gender: Male or Female

Weight (kg):
- Dogs: 2–60 (Chihuahua ~2kg, Beagle ~12kg, Labrador ~35kg, Rottweiler ~55kg)
- Cats: 2–8

Symptom_1 to Symptom_4: choose from this vocabulary (use 2-4 relevant ones,
leave unused slots as "None"):
Fever, Lethargy, Loss of Appetite, Coughing, Nasal Discharge, Vomiting,
Diarrhea, Sneezing, Weight Loss, Labored Breathing, Swollen Joints, Eye
Discharge, Skin Lesions, Pale Gums, Bloody Stool, Weakness, Seizures,
Excessive Thirst, Jaundice, Limping, Excessive Scratching, Hair Loss,
Bad Breath, Drooling, Bleeding, Difficulty Urinating, Increased Urination,
Swollen Abdomen, Head Shaking, Ear Scratching, Collapse

Duration: realistic string like "2 days", "1 week", "3 days", "2 weeks"

Appetite_Loss, Vomiting, Diarrhea, Coughing, Labored_Breathing, Lameness,
Skin_Lesions, Nasal_Discharge, Eye_Discharge: Yes or No

CRITICAL — Binary flag consistency rule:
These binary flags MUST match the Symptom_1-4 text fields.
- If any Symptom column = "Vomiting" → Vomiting = Yes
- If any Symptom column = "Coughing" → Coughing = Yes
- If any Symptom column = "Diarrhea" → Diarrhea = Yes
- If any Symptom column = "Labored Breathing" → Labored_Breathing = Yes
- If any Symptom column = "Lameness" or "Limping" → Lameness = Yes
- If any Symptom column = "Skin Lesions" or "Excessive Scratching" → Skin_Lesions = Yes
- If any Symptom column = "Nasal Discharge" → Nasal_Discharge = Yes
- If any Symptom column = "Eye Discharge" → Eye_Discharge = Yes
- If any Symptom column = "Loss of Appetite" → Appetite_Loss = Yes
A symptom in the text that is NOT reflected in the binary flag is a data error.

Body_Temperature (°C):
- Normal dog: 38.3–39.2
- Mild fever dog: 39.3–39.8
- High fever dog: 39.9–41.5
- Normal cat: 38.1–39.2
- Mild fever cat: 39.3–39.8
- High fever cat: 39.9–41.0

Heart_Rate (bpm):
- Normal dog: 60–140
- Elevated dog: 141–180
- Normal cat: 120–220
- Elevated cat: 221–260

Disease_Category: use exactly the category name specified in each prompt

Urgency — assign based on clinical severity:
- Emergency: immediate vet care needed (use for ~20% of rows)
  Signs: seizures, collapse, pale gums, bloody stool, labored breathing,
  suspected poisoning, temperature > 40.5°C, heart rate extremely abnormal
- Monitor: vet visit within 24-48 hours (use for ~50% of rows)
  Signs: persistent vomiting/diarrhea, moderate fever, not eating for 1+ day,
  progressive symptoms
- Okay: home monitoring, mild symptoms (use for ~30% of rows)
  Signs: single mild symptom, eating/drinking normally, alert and active,
  symptoms < 1 day

## EXAMPLE ROWS (follow this format exactly):

Dog,Labrador,4,Male,25.0,Fever,Lethargy,Vomiting,None,3 days,Yes,Yes,No,No,No,No,No,No,No,39.5,120,Gastrointestinal,Monitor
Cat,Siamese,2,Female,4.5,Coughing,Sneezing,Eye Discharge,Nasal Discharge,1 week,No,No,No,Yes,No,No,No,Yes,Yes,38.9,150,Respiratory,Okay
```

---

## CATEGORY PROMPTS

Run one at a time. Save each output to the filename shown.

---

### Category 1 — Respiratory
**Save to:** `data/raw/synthetic/respiratory.csv`

```
Generate 200 rows for Disease_Category = "Respiratory"

This category covers: Kennel Cough, Canine Influenza, Feline Calicivirus,
Feline Herpesvirus, Upper Respiratory Infection, Feline Asthma, Feline
Rhinotracheitis, Chronic Bronchitis, Allergic Rhinitis.

Key characteristics:
- Primary symptoms: Coughing, Nasal Discharge, Sneezing, Labored Breathing,
  Eye Discharge
- Feline Asthma → Labored Breathing is prominent, Urgency often Monitor/Emergency
- Kennel Cough → Dogs only, honking cough, usually Okay or Monitor
- Upper Respiratory → both dogs and cats, similar to human cold
- Mix: 60% cats, 40% dogs (respiratory issues are more common in cats)
- Temperature often normal to mild fever
- Urgency distribution: 15% Emergency, 45% Monitor, 40% Okay
```

---

### Category 2 — Gastrointestinal
**Save to:** `data/raw/synthetic/gastrointestinal.csv`

```
Generate 200 rows for Disease_Category = "Gastrointestinal"

This category covers: Canine Parvovirus, Gastroenteritis, Pancreatitis,
Canine Hepatitis, Feline Panleukopenia, Inflammatory Bowel Disease.

Key characteristics:
- Primary symptoms: Vomiting, Diarrhea, Loss of Appetite, Lethargy,
  Bloody Stool (in severe cases), Swollen Abdomen
- Parvovirus → young dogs (age 1-2), Bloody Stool, high fever, Emergency
- Pancreatitis → middle-aged dogs, after dietary indiscretion, severe vomiting
- Gastroenteritis → both dogs and cats, mild to moderate
- Mix: 60% dogs, 40% cats
- Urgency distribution: 30% Emergency (parvovirus/severe cases),
  45% Monitor, 25% Okay
```

---

### Category 3 — Viral Systemic
**Save to:** `data/raw/synthetic/viral_systemic.csv`

```
Generate 200 rows for Disease_Category = "Viral Systemic"

This category covers: Canine Distemper, Feline Infectious Peritonitis (FIP),
Feline Leukemia Virus (FeLV), Feline Immunodeficiency Virus (FIV).
(Note: Feline Panleukopenia is classified under Gastrointestinal, not here.)

Key characteristics:
- These are serious systemic diseases — generally high urgency
- Distemper → dogs, multi-system (respiratory + GI + neurological stages),
  Seizures possible in late stage
- FIP → cats, Swollen Abdomen, Weight Loss, chronic
- FeLV/FIV → cats, immunosuppression, Weight Loss, recurring infections
- Mix: 55% cats, 45% dogs
- Urgency distribution: 40% Emergency, 50% Monitor, 10% Okay
```

---

### Category 4 — Bacterial / Parasitic
**Save to:** `data/raw/synthetic/bacterial_parasitic.csv`

```
Generate 200 rows for Disease_Category = "Bacterial / Parasitic"

This category covers: Canine Leptospirosis, Lyme Disease, Intestinal
Parasites, Canine Heartworm Disease, Tick-Borne Disease.

Key characteristics:
- Leptospirosis → dogs, jaundice, kidney failure, Excessive Thirst,
  very high urgency
- Lyme Disease → dogs (tick exposure), Limping, Lethargy, Swollen Joints
- Intestinal Parasites → puppies/kittens, Weight Loss, Diarrhea, mild-moderate
- Heartworm → dogs mostly, Coughing, exercise intolerance, Labored Breathing
- Mix: 70% dogs, 30% cats
- Urgency distribution: 25% Emergency, 55% Monitor, 20% Okay
```

---

### Category 5 — Skin / Fungal
**Save to:** `data/raw/synthetic/skin_fungal.csv`

```
Generate 200 rows for Disease_Category = "Skin / Fungal"

This category covers: Ringworm, Fungal Infection, Allergic Dermatitis,
Mange, Hot Spots, Flea Allergy Dermatitis.

Key characteristics:
- Primary symptoms: Skin Lesions, Excessive Scratching, Hair Loss,
  Lethargy (in severe cases)
- Ringworm → circular lesions, Hair Loss, contagious
- Allergic Dermatitis → Excessive Scratching, Skin Lesions, chronic
- Temperature usually normal (unless secondary infection)
- Mix: 50% dogs, 50% cats
- Urgency distribution: 5% Emergency (severe secondary infections only),
  40% Monitor, 55% Okay
```

---

### Category 6 — Musculoskeletal
**Save to:** `data/raw/synthetic/musculoskeletal.csv`

```
Generate 200 rows for Disease_Category = "Musculoskeletal"

This category covers: Arthritis, Hip Dysplasia, Cruciate Ligament Injury,
Bone Fracture, Intervertebral Disc Disease (IVDD), Luxating Patella.

Key characteristics:
- Primary symptoms: Limping, Lameness, Swollen Joints, Weakness,
  reluctance to move
- Arthritis → older animals (age 7-15), chronic, gradual onset
- Fracture/Injury → any age, sudden onset, Collapse possible, higher urgency
- IVDD → Dachshunds especially, back pain, Weakness in hind legs
- Temperature normal unless infection
- Mix: 60% dogs, 40% cats
- Urgency distribution: 25% Emergency (fractures, acute disc),
  50% Monitor, 25% Okay (chronic arthritis management)
```

---

### Category 7 — Eye / Ear
**Save to:** `data/raw/synthetic/eye_ear.csv`

```
Generate 200 rows for Disease_Category = "Eye / Ear"

This category covers: Conjunctivitis, Corneal Ulcer, Glaucoma, Cataracts,
Ear Infection (Otitis), Ear Mites.

Key characteristics:
- Eye symptoms: Eye Discharge, squinting, redness (represent as Eye Discharge = Yes)
- Ear symptoms: Head Shaking, Ear Scratching (NOT Nasal Discharge — that is respiratory)
- Glaucoma → sudden pain, Emergency
- Ear Infections → very common, usually Monitor or Okay
- Corneal Ulcer → painful, needs prompt treatment
- Mix: 50% dogs, 50% cats
- Urgency distribution: 15% Emergency (glaucoma, severe ulcer),
  50% Monitor, 35% Okay
```

---

### Category 8 — Renal / Urinary
**Save to:** `data/raw/synthetic/renal_urinary.csv`

```
Generate 200 rows for Disease_Category = "Renal / Urinary"

This category covers: Urinary Tract Infection (UTI), Kidney Disease (CKD),
Urinary Blockage, Bladder Stones, Urinary Incontinence.

Key characteristics:
- UTI → Difficulty Urinating, Increased Urination, mild fever, common
- Kidney Disease → cats especially (older), Excessive Thirst, Weight Loss,
  Vomiting
- Urinary Blockage → CATS especially (male cats), unable to urinate,
  Swollen Abdomen, ALWAYS Emergency
- CKD → chronic, older cats and dogs
- Mix: 55% cats, 45% dogs
- Urgency distribution: 30% Emergency (blockages, acute kidney failure),
  50% Monitor, 20% Okay
```

---

### Category 9 — Endocrine / Metabolic
**Save to:** `data/raw/synthetic/endocrine_metabolic.csv`

```
Generate 200 rows for Disease_Category = "Endocrine / Metabolic"

This category covers: Diabetes Mellitus, Hyperthyroidism (cats),
Hypothyroidism (dogs), Cushing's Disease (dogs), Addison's Disease.

Key characteristics:
- Diabetes → Excessive Thirst, Increased Urination, Weight Loss despite
  normal appetite, both dogs and cats
- Hyperthyroidism → older cats, Weight Loss, hyperactivity, increased appetite
- Hypothyroidism → dogs, Weight gain, lethargy, cold intolerance
- Cushing's → dogs, pot-bellied appearance, Excessive Thirst
- Temperature usually normal
- Mix: 50% cats (thyroid), 50% dogs (diabetes, thyroid, cushing's)
- Urgency distribution: 15% Emergency (diabetic crisis), 60% Monitor,
  25% Okay (managed cases)
```

---

### Category 10 — Dental / Oral
**Save to:** `data/raw/synthetic/dental.csv`

```
Generate 200 rows for Disease_Category = "Dental / Oral"

This category covers: Periodontal Disease, Tooth Fracture, Oral Tumor,
Stomatitis (cats), Gingivitis, Tooth Root Abscess.

Key characteristics:
- Primary symptoms: Bad Breath, Loss of Appetite (pain when eating),
  Drooling (represent as Nasal Discharge = No, Eye Discharge = No,
  note drooling via Symptom columns)
- Stomatitis → cats, extremely painful mouth, Emergency
- Periodontal Disease → chronic, very common in older dogs and cats
- Tooth Abscess → facial swelling, fever, pain
- Mix: 50% dogs, 50% cats
- Urgency distribution: 20% Emergency (stomatitis, abscess with fever),
  45% Monitor, 35% Okay (chronic mild dental disease)
```

---

### Category 11 — Trauma / Poisoning
**Save to:** `data/raw/synthetic/trauma_poisoning.csv`

```
Generate 200 rows for Disease_Category = "Trauma / Poisoning"

This category covers: Hit by Car, Fall Injury, Animal Bite, Toxin Ingestion
(chocolate, grapes, xylitol, household chemicals, plants), Burns,
Heatstroke.

Key characteristics:
- This category has the HIGHEST urgency — most cases are Emergency
- Poisoning → Vomiting, Seizures, Pale Gums, Collapse, Bloody Stool
- HBC (hit by car) → Collapse, Labored Breathing, Pale Gums, Emergency
- Heatstroke → very high temperature (41+°C), Lethargy, Collapse, summer
- Animal Bite → Skin Lesions, pain, fever
- Mix: 55% dogs, 45% cats
- Urgency distribution: 65% Emergency, 30% Monitor (mild bites, minor
  ingestion), 5% Okay
- Temperature range: normal to 42°C (heatstroke)
```

---

## After Running All 11 Prompts

You'll have 11 CSV files in `data/raw/synthetic/`.
The next notebook (`03_llm_data_merge.ipynb`) will:
1. Load all 11 files
2. Validate schema consistency across files
3. Combine with the seed data from `data/processed/dogs_cats_labeled.csv`
4. Check for data quality issues
5. Save final training dataset to `data/processed/training_data.csv`
