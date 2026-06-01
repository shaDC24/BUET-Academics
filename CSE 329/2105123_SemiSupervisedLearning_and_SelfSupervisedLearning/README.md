# CSE 329: Machine Learning — Prompt Engineering Assignment
## Submission README

---

## Student Information

| Field | Details |
|---|---|
| **Name** | Shatabdi Dutta Chowdhury |
| **Student ID** | 2105123 |
| **Topic Number** | 24 |
| **Topic Name** | Semi-Supervised and Self-Supervised Learning |
| **LLM Used** | Gemini 2.5 Pro (Google AI Studio) |
| **LLM Version** | gemini-2.5-pro (accessed April 2025) |
| **Submission Date** | 11 April, 2026 |

---

## Topic Assignment Verification

```
Topic Number = (Student ID mod 25) + 1
             = (2105123 mod 25) + 1
             = 23 + 1
             = 24  ( Semi-Supervised and Self-Supervised Learning )
```

---

## Repository Structure

```
2105123_SemiSupervisedAndSelfSupervisedLearning/
│
├── README.md                           This file
│
├── prompts/                            Deliverable A: Modular Prompt System
│   ├── system_prompt.md                Master system prompt (role, constraints, meta-rules)
│   ├── module1.md                      Module 1 prompt: Lecture Note Generator
│   ├── module2.md                      Module 2 prompt: Theory Assessment Generator
│   └── module3.md                      Module 3 prompt: Coding Assessment Generator
│
├── outputs/                            Deliverable B: HITL-Reviewed Generated Content
│   ├── lecture_note.md                 Final lecture note (Parts A, B, C)
│   ├── theory_assessment.md            Final theory assessment 
│   ├── coding_boilerplate.py           Student-facing boilerplate (TODOs T1–T7)
│   └── coding_solution.py              Complete instructor solution with benchmarks
│
└── hitl_log/                           Deliverable C: Human-in-the-Loop Review Log
    └── hitl_review_log.md              Structured per-module HITL log 
```

---

## Approach Summary

### Prompt Engineering Techniques Used (≥ 4 required)

| # | Technique | Where Applied |
|---|-----------|---------------|
| 1 | **Role Prompting** | `system_prompt.md` — LLM assigned identity as "PedagogyGPT", an expert ML teaching assistant with explicit expertise attributes |
| 2 | **Chain-of-Thought (CoT)** | All three module prompts — explicit Step 1 planning section instructs the model to reason through structure before generating content |
| 3 | **Few-Shot Prompting** | `module2.md` — Good vs Bad Bloom's question examples provided; `module3.md` — exact TODO block format demonstrated with a complete worked example |
| 4 | **Output Structuring / Format Constraints** | All modules — precise Markdown section ordering, table formats, code block headers, and checklist templates specified |
| 5 | **Constraint Setting** | `system_prompt.md` — Universal Quality Standards enforce technical, pedagogical, ethics, and formatting rules globally across all modules |
| 6 | **Meta-Prompting** | `system_prompt.md` — Cross-Module Consistency Rules instruct the model to verify concept traceability and notation consistency before generating |

---

## HITL Process Summary

All LLM outputs were reviewed through the mandatory 5-step HITL workflow (Design → Accuracy → Pedagogy → Code Test → Finalize). A total of **9 issues** were identified and resolved across three modules.

| Module | Issues Found | Severity Distribution | Resolution |
|--------|-------------|----------------------|------------|
| Module 1 — Lecture Note | 5 | 2 HIGH, 2 MEDIUM, 1 LOW | 2 re-prompts + 3 manual edits |
| Module 2 — Theory Assessment | 3 | 2 HIGH, 1 MEDIUM | 3 manual rewrites |
| Module 3 — Coding Assessment | 4 | 1 CRITICAL, 2 HIGH, 1 MEDIUM | 3 manual fixes + 1 design decision |

Key corrections made:

- **M1:** Added missing BYOL and MAE explanations (HIGH); completed the SimCLR full pipeline beyond loss function only (HIGH); added formal pseudocode for Self-Training and Co-Training (MEDIUM); expanded ethical case study from ~180 to ~420 words (MEDIUM); completed 7-row comparison table (LOW)
- **M2:** Replaced recall question mislabeled as Apply with a concrete numerical Label Propagation trace (HIGH); replaced two Analyze questions that restated lecture note examples verbatim with novel domain scenarios (HIGH); redistributed marks per Bloom's-level specification (MEDIUM)
- **M3:** Fixed device mismatch bug in NT-Xent loss targets causing `RuntimeError` on CUDA (CRITICAL); kept augmentation as pre-completed with understanding requirement rather than an ambiguous free-form TODO (decision); added missing benchmark block to solution file (MEDIUM); clarified `evaluate_and_compare` return type (MEDIUM)

Full documentation for every issue — raw output, correction applied, and rationale — is in `hitl_log/hitl_review_log.md`.

---

## Code Execution

### Requirements

```
Python 3.10+
numpy==1.24.x
scikit-learn==1.3.x
torch==2.1.x
torchvision==0.16.x
matplotlib==3.7.x
```

Install dependencies:
```bash
pip install numpy==1.24.4 scikit-learn==1.3.2 torch==2.1.2 torchvision==0.16.2 matplotlib==3.7.5
```

### Running the Boilerplate (Students)

```bash
python outputs/coding_boilerplate.py
```

Expected: All TODOs raise `NotImplementedError`. The `validate_student_work()` harness can be called independently at any point to check individual implementations.

### Running the Solution (Instructor Verification)

```bash
python outputs/coding_solution.py
```

---

## Academic Integrity Declaration

- LLM-generated content was used as the assignment specifies — as raw material subject to HITL review, not as final output.
- All outputs were reviewed through a genuine HITL process; the review log was written during review, not reconstructed after the fact.
- The LLM model, version, and platform are disclosed above.


---

*CSE 329: Machine Learning | Prompt Engineering Assignment | Topic 24*
