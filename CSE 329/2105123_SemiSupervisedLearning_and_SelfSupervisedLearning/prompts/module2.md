# Module 2 Prompt — Theory Assessment Generator


---

## Context

You have already generated a lecture note for **Semi-Supervised and Self-Supervised Learning** (Module 1). The key concepts covered were:

- Motivation and problem setting (labeled data bottleneck)
- SSL assumptions (smoothness, cluster, manifold)
- GMM + EM algorithm for semi-supervised learning
- Label Propagation (graph-based)
- Self-Training and Co-Training
- Self-supervised pretext tasks
- Contrastive learning (InfoNCE loss, SimCLR)
- Non-contrastive methods (BYOL, MAE)
- Ethical risks (bias amplification, privacy, societal impact)

Your assessment **must test exactly these concepts** — no concept should appear in the assessment that was not taught in the lecture note. This ensures constructive alignment.

---

## Step 1 — Chain-of-Thought Planning (Internal, Do Not Print)

Before writing questions, think through:

1. For each Bloom's level (1–6), what aspect of SSL/Self-SL is best tested at that level?
   - Remember: definitions, formulas, algorithm steps
   - Understand: explain *why* something works
   - Apply: compute something, trace an algorithm
   - Analyze: diagnose a failure, compare methods
   - Evaluate: defend a design choice, critique a method
   - Create: design a new system, propose an experiment

2. For levels 4–6, ensure the scenarios are **novel** — not examples from the lecture note restated.

3. Which Evaluate or Create question will engage with an ethical dimension?

4. Are there questions that require multi-step reasoning (chain-of-thought answers)?

Use this planning to write balanced, well-distributed questions. Do not print the planning.

---

## Step 2 — Few-Shot Examples of Question Quality

**Example of a GOOD Level-1 (Remember) question:**
> **Q:** Define the smoothness assumption in semi-supervised learning.
> **Bloom's Level:** Remember | **Action Verb:** Define
> **Answer Key:** If two points $x_i$ and $x_j$ are close in input space, their labels $y_i$ and $y_j$ should also be similar. [1 mark for correct statement, 1 mark for mathematical or intuitive elaboration]

**Example of a BAD Level-4 (Analyze) question (DO NOT DO THIS):**
> ~~**Q:** What is the EM algorithm?~~ ← This is Remember, not Analyze.

**Example of a GOOD Level-4 (Analyze) question:**
> **Q:** A practitioner runs semi-supervised GMM+EM on a dataset with two classes. After 50 iterations, the model's accuracy is *worse* than the supervised baseline trained on labeled data alone. Identify two possible causes of this failure and explain the mechanism behind each.
> **Bloom's Level:** Analyze | **Action Verb:** Diagnose

Use these examples as quality benchmarks. Every question you generate must meet or exceed this standard.

---

## Step 3 — Generate the Theory Assessment

Produce the assessment in **exactly** this structure. Label every question clearly.

---

# Theory Assessment: Semi-Supervised and Self-Supervised Learning
### CSE 329: Machine Learning | Total: [auto-calculate from marks]

---

## Section 1 — Remember (Recall)
*Minimum 2 questions. Action verbs: Define, List, State, Recall, Identify.*

For each question, use this format:

---
**Q[number] | Bloom's Level: Remember | Action Verb: [verb] | Marks: [X]**

[Question text]

**Answer Key:**
- Model Answer: [complete answer]
- Marking Scheme: [how marks are distributed, including partial credit rules]
- Expected Response Depth: [1–2 sentences for Remember level]
---

## Section 2 — Understand (Comprehension)
*Minimum 2 questions. Action verbs: Explain, Compare, Summarize, Interpret.*

[Same format as above]

## Section 3 — Apply (Procedural Use)
*Minimum 2 questions. Action verbs: Calculate, Implement, Solve, Predict, Trace.*

[Same format. At least one question must involve a numerical calculation or algorithm trace.]

## Section 4 — Analyze (Decomposition)
*Minimum 2 questions. Scenarios must be NOVEL — not from the lecture note.*
*Action verbs: Differentiate, Contrast, Diagnose, Examine, Deconstruct.*

[Same format. Questions must present a new scenario for the student to reason about.]

## Section 5 — Evaluate (Judgment)
*Minimum 2 questions. At least ONE must engage with an ethical dimension.*
*Action verbs: Justify, Critique, Assess, Argue, Recommend.*

[Same format. Ethical question must be specific to SSL/Self-SL mechanisms — not generic AI ethics.]

## Section 6 — Create (Synthesis)
*Minimum 2 questions. Must require original design or synthesis.*
*Action verbs: Design, Propose, Formulate, Synthesize, Construct.*

[Same format. Questions should be open-ended with rubric-based marking.]

---

## Assessment Summary Table

After all questions, produce:

| Q# | Section | Bloom's Level | Action Verb | Topic Area | Marks |
|----|---------|---------------|-------------|------------|-------|
| Q1 | 1 | Remember | Define | SSL assumptions | X |
| ... | | | | | |
| **Total** | | | | | **[sum]** |

---

## Step 4 — Alignment Verification

After the summary table, output:

```
ALIGNMENT CHECK:
- Every question maps to a concept in the lecture note: [YES/NO — list any gaps]
- Bloom's levels correctly applied (no recall questions mislabeled as analysis): [YES/NO]
- At least one Evaluate/Create question addresses ethics: [YES/NO — which question(s)]
- Novel scenarios used at levels 4–6: [YES/NO — confirm for each question]
- All answer keys include partial credit rules: [YES/NO]
```

---

## Constraints

- Minimum 12 questions total (2 per level × 6 levels).
- Do not add more than 4 questions at level 1 or 2 — the assessment must emphasize higher-order thinking.
- Every question must have a unique number (Q1, Q2, ...).
- No question at level 4, 5, or 6 may restate an example directly from the lecture note.
- Marks per question: Remember = 2–3 marks, Understand = 3–4 marks, Apply = 4–5 marks, Analyze = 5–6 marks, Evaluate = 6–8 marks, Create = 8–10 marks.
- Answer keys for levels 5–6 must include a sample high-scoring response and explain why it earns full marks.
