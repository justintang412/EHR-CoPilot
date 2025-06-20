# Building a Clinical-Grade EHR Assistant

## What Is EHR Assistant

In a typical clinical encounter, a physician follows a structured process:
1. Records the patient's "Chief Complaint" (CC) - the primary reason for the visit
2. Documents the "History of Present Illness" (HPI) - comprehensive details about current symptoms and their progression
3. Conducts physical examination and reviews relevant laboratory results
4. Formulates an "Assessment, Procedure and Plan" (APP)
5. Issues appropriate prescriptions, laboratory orders, and follow-up appointments

Given the capabilities of modern Large Language Models (LLMs), one might wonder: Can we simply input all patient information into an LLM to generate a well-structured clinical assessment?

Probabily not.

We need more:

1. Precision and Reliability
LLMs are probabilistic and can introduce critical errors in clinical contexts:
   - Misinterpret ambiguous abbreviations (e.g., "CP" = chest pain or cerebral palsy?)
   - Miss subtle temporal cues ("pain worse in AM vs PM")
   - Overlook negations ("no chest pain" ≠ "chest pain")

1. Explainability & Auditing
Clinical-grade tools must provide clear reasoning for their suggestions:
   - Raw LLM outputs lack traceability and auditability
   - Structured data (e.g., symptom: chest pain, duration: 2 days) enables logging, visualization, and validation
   - Structured preprocessing enables transparent, reproducible clinical reasoning

1. Regulatory & Compliance (HIPAA, FDA SaMD)
Clinical decision tools may be classified as Software as a Medical Device (SaMD), requiring:
   - Structured input-output logs
   - Deterministic behavior
   - Comprehensive error analysis of components
   - Explicit intermediate processing steps for traceability

1. Data Quality & Variability
EHR notes present unique challenges:
   - Copied/pasted sections
   - Irrelevant information (billing codes, template noise)
   - Mixed temporal events (old vs current complaints)

1. Interoperability
Clinical systems must integrate with existing healthcare infrastructure:
   - Structured triage scores
   - Clinical alerts
   - Care plan templates
   - FHIR APIs and hospital systems

## System Architecture

### Six Core Layers

Our EHR Assistant consists of six core layers:

1. **Input Processing**
   - Cleans and normalizes clinical text from EHR
   - Handles basic text corrections

2. **Clinical NLP Pipeline**
   - Splits notes into standard clinical sections
   - Identifies medical terms, symptoms, and medications
   - Detects negations and temporal information
   - Maps terms to medical standards (SNOMED CT, ICD-10)
   - Removes sensitive patient information

3. **Patient Context**
   - Combines current visit data with historical records
   - Creates a complete patient profile

4. **AI Reasoning**
   - Converts structured data into LLM prompts
   - Uses advanced language models for clinical analysis
   - Applies safety checks and medical guidelines

5. **Output Processing**
   - Generates structured outputs:
     - Top diagnoses
     - Lab recommendations
     - Treatment plans
     - Clinical summaries

6. **System Integration**
   - Connects with EHR systems
   - Provides audit logs
   - Enables quality monitoring
  
Here is an example of how notes go through the layers:

**Sample Clinical Note**
```
Patient complains of chest pain for the last 2 days. Denies fever or vomiting. Pain started after exercise, radiates to the left arm. PMH includes hypertension. No allergies.
```

**How the Note Flows Through Each Layer**

1. **Input Processing**
   - Cleans and normalizes the text:
   ```
   Sentence 1: Patient complains of chest pain for the last 2 days.
   Sentence 2: Denies fever or vomiting.
   Sentence 3: Pain started after exercise, radiates to the left arm.
   Sentence 4: PMH includes hypertension.
   Sentence 5: No allergies.
   ```

2. **Clinical NLP Pipeline**
   - Splits into clinical sections:
   ```json
   {
     "chief_complaint": "chest pain",
     "history_present_illness": "chest pain for the last 2 days. Pain started after exercise, radiates to the left arm.",
     "review_of_systems": "Denies fever or vomiting.",
     "past_medical_history": "hypertension",
     "allergies": "No allergies"
   }
   ```
   - Identifies medical terms:
   ```json
   [
     {"text": "chest pain", "type": "symptom"},
     {"text": "fever", "type": "symptom"},
     {"text": "vomiting", "type": "symptom"},
     {"text": "hypertension", "type": "condition"},
     {"text": "left arm", "type": "anatomical site"}
   ]
   ```
   - Detects negations and timing:
   ```json
   {
     "negated_symptoms": ["fever", "vomiting"],
     "temporal_info": {
       "symptom": "chest pain",
       "duration": "2 days",
       "onset_trigger": "after exercise"
     }
   }
   ```
   - Maps to medical standards:
   ```json
   [
     {"term": "chest pain", "snomed": "29857009"},
     {"term": "hypertension", "snomed": "38341003"}
   ]
   ```

3. **Patient Context**
   - Combines with historical data:
   ```json
   {
     "current_visit": {
       "symptoms": ["chest pain"],
       "negated_symptoms": ["fever", "vomiting"],
       "duration": "2 days",
       "trigger": "after exercise",
       "site": "left arm"
     },
     "past_records": {
       "conditions": ["hypertension"],
       "medications": ["lisinopril"],
       "recent_labs": [{"test": "cholesterol", "value": "220", "unit": "mg/dL"}]
     }
   }
   ```

4. **AI Reasoning**
   - Generates clinical prompt:
   ```
   A patient with hypertension presents with chest pain for 2 days, triggered by exercise, radiating to the left arm. No fever or vomiting. What are the top possible diagnoses, recommended tests, and treatment plan?
   ```
   - Produces structured analysis:
   ```json
   {
     "differential_diagnoses": [
       "Acute Coronary Syndrome",
       "Stable Angina",
       "Musculoskeletal Chest Pain"
     ],
     "recommended_tests": [
       "ECG",
       "Troponin"
     ],
     "treatment_plan": [
       "Administer aspirin",
       "Cardiology consult"
     ]
   }
   ```

5. **Output Processing**
   - Formats final structured output:
   ```json
   {
     "diagnoses": ["ACS", "Angina", "Musculoskeletal pain"],
     "tests": ["ECG", "Troponin"],
     "treatment": ["Aspirin", "Cardiology consult"],
     "urgency": "High (possible cardiac event)",
     "clinical_summary": "58-year-old with hypertension presenting with 2-day history of exercise-induced chest pain radiating to left arm. No associated fever or vomiting. High suspicion for cardiac etiology."
   }
   ```

6. **System Integration**
   - Sends structured output to:
     - EHR system for documentation
     - Clinical decision support tools
     - Quality assurance logs
   - Records interaction for audit and improvement

### Fine-Tuning Plan

<blockquote>
  <p>"If BioClinicalBERT can extract symptoms with 90% F1, don't train a new model just to improve by 2%."</p>
  <cite>— Real quote from a clinical NLP team at a major healthtech company.</cite>
</blockquote>

While pretrained models provide excellent baselines, clinical applications often require fine-tuning to handle domain-specific challenges. Here's a strategic approach to model training:

#### 1. Fine-tune NER (Symptoms, Meds, Complaints)
**Why:** Pretrained models often miss abbreviations, shorthand, or clinic-specific terms.

**Start From:** BioClinicalBERT or roberta-base-bionlp-ner

**What to Do:** Fine-tune on labeled spans (BIO format) for symptoms, medications, durations, etc.

**Data Needed:** 1,000–5,000 annotated sentences

**Effort:** Medium — can be done with Hugging Face Trainer or spaCy custom NER.

#### 2. Train Relation Extractor (onset, symptom→duration, etc.)
**Why:** Needed to convert entity pairs into relationships the LLM or rule engine can reason on.

**Start From:** roberta-base or clinicalBERT for pairwise classification

**What to Do:** Supervised classification: Entity1 + Entity2 + context → relation_type

**Data Needed:** 2,000–10,000 entity pairs with labels

**Effort:** Medium-high. May need annotation interface like Prodigy or Doccano

#### 3. Train Symptom-to-Diagnosis or Triage Classifier (optional backup)
**Why:** Build a rule-based or ML backup to LLMs for explainability, redundancy.

**Start From:** LightGBM, XGBoost, or TabTransformer with structured features

**What to Do:** Use extracted symptoms, vitals, and history as input → classify diagnosis group or urgency

**Data Needed:** 5,000–50,000 structured cases

**Effort:** Low-medium, assuming structured features are available or can be engineered

#### 4. Train Prompt Selector / Generator (if needed)
**Why:** To choose the right prompt template or examples based on patient data complexity.

**Start From:** MiniLM or sentence-transformers

**What to Do:** Build few-shot prompt embedding index + similarity search, or train a small classifier.

**Data Needed:** Few hundred–few thousand prompt-context pairs

**Effort:** Low, but optional

#### 5. (Optional) Domain-Adaptive LM Pretraining
**Why:** To boost all downstream tasks if your clinic's data is different (e.g., shorthand-heavy).

**Start From:** BioClinicalBERT or PubMedBERT

**What to Do:** Continue pretraining on your unlabeled notes (masked LM objective)

**Data Needed:** 10,000+ unlabeled notes (ideal: >1M tokens)

**Effort:** High. Requires GPU, good tokenizer, and patience.

**Only do this if:** Multiple models are underperforming on your data.

## How to Measure Success

Measuring accuracy in a clinical-grade assistant requires evaluation across multiple layers, each serving different clinical purposes. Here's a comprehensive framework for evaluating your EHR assistant system:

### 1. Named Entity Recognition (NER)
**Purpose:** Extracting symptoms, conditions, medications, durations, and anatomical sites

**Metrics:**
- **Precision:** Percentage of extracted entities that are correct
- **Recall:** Percentage of all true entities that were extracted  
- **F1 Score:** Harmonic mean of precision and recall

**Tools:**
- `seqeval` (Python)
- Hugging Face `evaluate` module
- Scikit-learn (for post-processed label lists)

**Example:**
```python
true = [['O', 'O', 'B-SYMPTOM', 'I-SYMPTOM']]
pred = [['O', 'O', 'B-SYMPTOM', 'O']]
# Calculate precision, recall, F1 for symptom extraction
```

**Clinical Threshold:** F1 score > 90%

### 2. Relation Extraction & Clinical Context
**Purpose:** Detecting relationships between entities (symptom→duration, negations, temporal information)

**Metrics:**
- **Precision/Recall/F1** for relation types:
  - Symptom → onset trigger
  - Symptom → duration
  - Negation detection (positive vs. negated symptoms)
  - Temporal relationships

**Example:**
```json
True:  ("chest pain", "triggered by", "exercise")
Pred:  ("chest pain", "triggered by", "exercise") (Correct)
```

**Clinical Threshold:** Accuracy/F1 > 95% for negation detection

### 🧠 3. Clinical Reasoning & Diagnosis Suggestion (LLM Layer)
**Purpose:** Generating differential diagnoses and treatment plans

**Metrics:**
- **Top-N Accuracy:** True diagnosis appears in top-1, top-3, or top-5 suggestions
- **Exact Match:** Direct match with ground truth diagnosis
- **Code Match:** ICD-10 or SNOMED CT code alignment
- **BLEU/ROUGE/METEOR:** For free-text plan generation quality
- **Clinical Concordance:** Alignment with physician consensus

**Example:**
```
True: Acute Coronary Syndrome
LLM: 1. Acute Coronary Syndrome (Correct)
     2. Stable Angina
     3. Musculoskeletal Pain
```

**Clinical Threshold:** Top-3 accuracy > 80-90%

### 4. End-to-End System Evaluation
**Purpose:** Assessing complete clinical workflow accuracy

**Metrics:**
- **Case-Level Accuracy:** Complete and accurate production of:
  - Extracted clinical findings
  - Suggested differential diagnoses
  - Safe treatment recommendations
  - Appropriate urgency assessment

- **Error Rate by Category:**
  - **Critical:** Missed red flags (dangerous)
  - **Moderate:** Wrong suggestions (harmless but incorrect)
  - **Minor:** Hallucinated information (invented past diagnoses)

**Manual Review Process:**
1. Sample 50-100 real clinic notes
2. Run through complete system pipeline
3. Have clinical reviewers label outcomes:
   - "Fully correct"
   - "Partially helpful" 
   - "Incorrect or misleading"

**Clinical Threshold:** Case-level match rate > 90%

### 5. LLM-Specific Evaluation
**Purpose:** Assessing quality of AI-generated clinical outputs

**Metrics:**
- **Graded Scores:** Likert scale (1-5) for:
  - Clinical helpfulness
  - Safety and appropriateness
  - Correctness and completeness

- **LLM-as-a-Judge:** Automated evaluations using GPT-4 with gold standard comparison
- **EMR Outcome Match:** Comparison with actual clinical outcomes (when available)

**Example Evaluation Prompt:**
```
Rate the following clinical assessment on a scale of 1-5:
- Helpfulness: How useful is this for clinical decision-making?
- Safety: Are there any dangerous or inappropriate suggestions?
- Correctness: Does this align with standard clinical practice?
```

### Summary Table

**NER Component:** Primary metric is F1 Score with clinical threshold > 90%, evaluated using seqeval and Hugging Face tools.

**Negation Detection:** Primary metric is Accuracy/F1 with clinical threshold > 95%, evaluated using classification metrics.

**Diagnosis Suggestion:** Primary metric is Top-3 Accuracy with clinical threshold > 80-90%, evaluated using exact match and code match methods.

**End-to-End Triage:** Primary metric is Case-level Match Rate with clinical threshold > 90%, evaluated through manual clinical review.

**LLM Output Quality:** Primary metric is Human Rating/GPT Eval with requirement for high consistency, evaluated using Likert scale and automated evaluation methods.

