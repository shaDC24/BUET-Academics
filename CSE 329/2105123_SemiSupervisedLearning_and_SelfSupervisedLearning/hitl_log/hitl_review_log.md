# HITL Review Log — All Modules
## CSE 329 | Topic: Semi-Supervised and Self-Supervised Learning
## Reviewer: Shatabdi Dutta Chowdhury | LLM: Gemini 2.5 Pro (Modules 1–3)

---

# MODULE 1 REVIEW — Lecture Note

## Raw Output Summary
The raw output covered Parts A, B, and C of the lecture note as required, but with five identified issues documented below. The final `Lecture_Note.md` incorporates all corrections.

---

### ISSUE-01 | Severity: HIGH | Section: A3.3

**Type:** Missing Section — Non-Contrastive Methods (BYOL and MAE)

**Prompt Requirement:**
> "A3.3 Non-Contrastive Methods — BYOL: explain momentum encoder, stop-gradient, why no collapse. MAE: masking ratio, reconstruction objective, encoder-decoder design."

**What Was Generated:**
Section A3.3 was titled "Comparison Table" but contained neither BYOL nor MAE explanations — only the comparison table with 5 rows (missing Self-Training, Co-Training, BYOL rows).

**Correction Required:**
Re-prompted Gemini with targeted instruction (see Re-Prompt block below). Added full BYOL explanation including momentum update equation $\xi \leftarrow m\xi + (1-m)\theta$, BYOL loss with stop-gradient $\mathcal{L}_{\text{BYOL}} = \|\bar{q}_\theta(z_i) - \text{sg}(\bar{z}'_j)\|^2$, collapse prevention mechanism, and full MAE explanation including asymmetric encoder-decoder, 75% masking ratio, and reconstruction loss $\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|}\sum_{p \in \mathcal{M}}\|x_p - \hat{x}_p\|^2$.

**Rationale:** BYOL and MAE are fundamental methods in the Self-SL landscape and are explicitly listed in the assignment topic. The comparison table alone is insufficient for a lecture note — students cannot implement or reason about methods they don't understand mechanistically. Module 2 will also test these concepts (per the consistency check), so this gap must be filled.

---

### ISSUE-02 | Severity: HIGH | Section: B3

**Type:** Incomplete Implementation — Core Deliverable Missing

**Prompt Requirement:**
> "Produce complete code that: defines the augmentation pipeline, defines the encoder (ResNet-18) + projection head (2-layer MLP), implements the NT-Xent loss, runs a short pretraining loop on CIFAR-10, evaluates with linear probe."

**What Was Generated:**
Only the NT-Xent loss function was implemented (~25 lines). The augmentation pipeline, ResNet-18 encoder, projection head, pretraining loop, and linear probe evaluation were all missing. This is approximately 20% of the required code.

**Correction Required:**
Re-prompted Gemini for the missing SimCLR pipeline. Added: `SimCLRAugmentation` class with full torchvision pipeline, `SimCLRModel` class with ResNet-18 backbone (CIFAR-adapted) + 2-layer MLP projector, a 5-epoch pretraining loop, and linear probe evaluation with accuracy reporting.

**Additional Bug Found (Manual Review):** The generated code used `datasets.CIFAR_10(...)` (with underscore) which is incorrect — the correct class name is `datasets.CIFAR10`. This was corrected in the final output.

**Rationale:** The full pipeline is the central pedagogical demonstration of the lecture note. A loss function in isolation cannot show students how augmentation strategy, encoder architecture, and evaluation interact.

---

### ISSUE-03 | Severity: MEDIUM | Section: A2.4

**Type:** Missing Formal Pseudocode — Pedagogical Gap

**Prompt Requirement:**
> "Self-training pseudocode and analysis. Co-training (Blum & Mitchell, 1998): two-view assumption, algorithm."

**What Was Generated:**
Both methods described in 2–3 sentences with no pseudocode. Co-training has no formal algorithm.

**Correction Required:**
Manually added formal pseudocode blocks for both Self-Training (with threshold δ, confident subset S, and re-training loop) and Co-Training (with two-view assumption, dual classifiers f¹ and f², and mutual labeling step).

**Rationale:** Pseudocode is explicitly required by the module prompt. Students need a precise, implementable description to connect theory to Module 3 code tasks.

---

### ISSUE-04 | Severity: MEDIUM | Section: C5

**Type:** Insufficient Depth — Word Count Below Requirement

**Prompt Requirement:**
> "Write a detailed case study (minimum 300 words)."

**What Was Generated:**
Approximately 180 words — well below the 300-word minimum. Missing: detection mechanisms and responsible alternatives beyond a single generic suggestion.

**Correction Required:**
Manually expanded C5 with two structured subsections: "How to Detect This Failure in Practice" (performance disaggregation, embedding space analysis via t-SNE/UMAP, calibration check via ECE) and "Responsible Alternatives" (Domain-Adaptive SSL, Federated Semi-Supervised Learning, HITL deployment, Uncertainty Quantification with MC Dropout). Final case study is approximately 420 words.

**Rationale:** The assignment rubric requires "substantive, topic-specific ethical analysis." Detection and mitigation strategies tied to specific SSL mechanisms distinguish a topic-specific case study from generic AI ethics.

---

### ISSUE-05 | Severity: LOW | Section: A3.3 Comparison Table

**Type:** Missing Rows and Column — Incomplete Table

**Prompt Requirement:**
> "Produce a table comparing: GMM+EM, Label Propagation, Self-Training, Co-Training, SimCLR, BYOL, MAE across: type, requires labels, scalability, typical domain, key assumption."

**What Was Generated:**
Table had 5 rows (missing Self-Training, Co-Training, BYOL) and 4 columns (missing "Requires Labels" column).

**Correction Required:**
Manually replaced with complete 7-row, 6-column table including all methods and the "Requires Labels" column.

**Rationale:** The prompt explicitly listed 7 methods and 5 columns. This is a direct completeness gap. The table is a likely source of assessment questions in Module 2.

---

## What Was Good (No Changes Needed) — Module 1

- **A1:** Formal definitions mathematically correct. Comparison table (Supervised vs Semi vs Self vs Unsupervised) is clean and accurate.
- **A2.2:** GMM log-likelihood formula correct. EM E-step formula $\gamma_{iy}$ correct and well-explained.
- **A2.3:** Label Propagation similarity formula and update rule correct.
- **A3.2:** InfoNCE loss formula correct. Notation ($z_i, z_j, \tau$) consistent.
- **B2:** Label Propagation code complete and runnable. Use of `-1` as unlabeled sentinel in sklearn correct.
- **B3 (partial):** NT-Xent loss implementation logic correct (cosine similarity matrix, masking, cross-entropy formulation).
- **B4:** Pitfalls table accurate and pedagogically useful.
- **B5:** Hyperparameter table accurate. Note about 75% masking ratio for vision is correct.
- **C1–C4:** Bias amplification explanation, privacy risk (membership inference), and responsible practices checklist all topic-specific and accurate.

---

## Decision: Re-Prompt or Manual Edit? — Module 1

| Issue | Action |
|---|---|
| ISSUE-01 | **Re-prompted** Gemini with targeted instruction for BYOL and MAE only |
| ISSUE-02 | **Re-prompted** Gemini for complete SimCLR pipeline; bug-fixed CIFAR_10 → CIFAR10 manually |
| ISSUE-03 | **Manual edit** — added pseudocode blocks directly to output file |
| ISSUE-04 | **Manual edit** — expanded C5 with two structured subsections |
| ISSUE-05 | **Manual edit** — replaced table in A3.3 with complete version |

---

# MODULE 2 REVIEW — Theory Assessment

## Raw Output Summary
The theory assessment was generated by running `module2.md` (with the lecture note consistency context) through Gemini 2.5 Pro. The raw output produced 12 questions across 6 Bloom's levels with answer keys. Three issues were identified and corrected.

---

### ISSUE-M2-01 | Severity: HIGH | Section: Q5 (Apply — Label Propagation trace)

**Type:** Missing Numerical Example — Apply Question Without Calculation

**Raw Output:**
Q5 asked students to "trace one iteration of label propagation" but provided no concrete numerical matrix — only the abstract update rule $F \leftarrow \hat{W}F$ restated from the lecture note. This is a Remember-level question mislabeled as Apply.

**Correction Required:**
Completely replaced Q5 with a concrete 4-node toy graph problem with explicit weight matrix $\hat{W}$ and initial label matrix $F^{(0)}$. Students must perform actual matrix multiplication, then apply clamping. A full numerical answer key with step-by-step arithmetic was added.

**Rationale:** The module prompt's Bloom's Table states Apply requires "Calculate, Implement, Solve, Predict, Trace" — a trace question without numbers to trace is not Apply-level. The assignment brief also states "At least one question at Apply level must involve a numerical calculation or algorithm trace."

---

### ISSUE-M2-02 | Severity: HIGH | Section: Q7 and Q8 (Analyze — Novel Scenarios)

**Type:** Lecture Note Examples Restated — Violates Novel Scenario Requirement

**Prompt Requirement:**
> "Questions at levels 4–6 must present novel scenarios — not lecture note examples restated."

**Raw Output:**
Q7 asked "Why does GMM+EM fail when generative assumptions are wrong?" — this is verbatim from lecture note section A2.2. Q8 asked "Compare Self-Training and Label Propagation" — identical to the comparison discussion in A2.4.

**Correction Required:**
- Q7 replaced with satellite imagery SimCLR failure scenario (novel domain not in lecture note), requiring diagnosis of two specific failure mechanisms with evidence linkage.
- Q8 replaced with a 10,000-image dataset scale analysis requiring three concrete trade-off factors (scalability of graph construction, classifier flexibility, iterative label control) — not a generic comparison.

**Rationale:** Analyze-level questions require students to decompose a novel problem, not recall a known failure mode. The lecture note example restated as a question is, at best, an Understand-level question.

---

### ISSUE-M2-03 | Severity: MEDIUM | Section: Marks Distribution

**Type:** Marks Too Uniform — All Questions Weighted Equally

**Raw Output:**
All 12 questions were marked at 5 marks each (total: 60 marks). The module prompt specifies: "Remember = 2–3 marks, Understand = 3–4 marks, Apply = 4–5 marks, Analyze = 5–6 marks, Evaluate = 6–8 marks, Create = 8–10 marks."

**Correction Required:**
Redistributed marks according to the Bloom's-level scale specified in the prompt:
- Q1 (Remember): 3 marks
- Q2 (Remember): 2 marks
- Q3 (Understand): 4 marks
- Q4 (Understand): 4 marks
- Q5 (Apply): 5 marks
- Q6 (Apply): 4 marks
- Q7 (Analyze): 6 marks
- Q8 (Analyze): 5 marks
- Q9 (Evaluate): 7 marks
- Q10 (Evaluate): 6 marks
- Q11 (Create): 10 marks
- Q12 (Create): 8 marks
- **Total: 64 marks**

**Rationale:** Uniform marks signal to students that all cognitive levels are equally valued, contradicting the assessment design philosophy that higher-order thinking (Evaluate, Create) requires more sustained effort and deserves more marks.

---

## What Was Good (No Changes Needed) — Module 2

- **Q1 (Remember/Define):** Smoothness, Cluster, Manifold definitions accurate. Marking scheme with partial credit for vague definitions is pedagogically appropriate.
- **Q2 (Remember/State):** InfoNCE formula correctly reproduced. Four terms (z_i, z_j, z_k, τ) all identified.
- **Q3 (Understand/Explain):** GMM+EM failure explanation correctly references the E-step corruption mechanism and the dominance of the unlabeled term. Well-structured 4-part marking scheme.
- **Q4 (Understand/Compare):** SimCLR vs BYOL comparison along collapse prevention, batch size dependency, and computational cost is accurate and pedagogically rich.
- **Q6 (Apply/Calculate):** InfoNCE loss numerical calculation is correctly set up. Cosine similarities and exponentials are correct.
- **Q9 (Evaluate/Critique):** Medical dermatology scenario is novel (not in lecture note). Three-part structure (pretraining data, contrastive objective suitability, responsible alternative) is strong and topic-specific. Correctly identifies augmentation-domain mismatch as a SSL-specific risk.
- **Q10 (Evaluate/Justify):** "One condition correct, one condition incorrect" format is an excellent Evaluate-level scaffold. Confirmation bias and manifold assumption references are accurate.
- **Q11 (Create/Design):** Wildlife edge device scenario is novel and requires genuine synthesis across Self-SL and Semi-SL methods. Four-part rubric is well-structured.
- **Q12 (Create/Propose):** EHR co-training system design correctly identifies the independence assumption verification challenge and provides a specific failure mode (lab values appear in both views).
- **Alignment Check block:** Correctly maps every question to a lecture note concept. No phantom concepts tested.

---

## Decision: Re-Prompt or Manual Edit? — Module 2

| Issue | Action |
|---|---|
| ISSUE-M2-01 | **Manual rewrite** — Q5 fully replaced with numerical toy graph problem and step-by-step answer key |
| ISSUE-M2-02 | **Manual rewrite** — Q7 and Q8 replaced with novel scenarios; answer keys rewritten |
| ISSUE-M2-03 | **Manual edit** — marks redistributed per Bloom's-level specification; summary table updated |

---

# MODULE 3 REVIEW — Coding Assessment

## Raw Output Summary
The coding assessment was generated by running `module3.md` through Gemini 2.5 Pro. Two Python files were produced: `coding_boilerplate.py` and `coding_solution.py`. Both files were manually tested by running them locally. Four issues were identified.

---

### ISSUE-M3-01 | Severity: CRITICAL | Section: Both files — `nt_xent_loss`

**Type:** Incorrect Loss Implementation — Wrong Positive Pair Labels

**What Was Generated:**
```python
# Raw output (incorrect):
targets = torch.arange(2 * batch_size)
targets[:batch_size] += batch_size
targets[batch_size:] -= batch_size
```
This mutates `targets` in-place after creation in a way that produces the correct intent but is fragile — if `targets` is on a different device than `sim_matrix`, `F.cross_entropy` raises a device mismatch error with no helpful message.

**Correction Required:**
Replaced with explicit concatenation ensuring both tensors are on the correct device from creation:
```python
labels = torch.cat([
    torch.arange(N, 2 * N, device=z_i.device),
    torch.arange(0, N, device=z_i.device)
])
```

**Test Result:** With the fix, running `nt_xent_loss(z_i, z_j)` on CPU produces loss ≈ 5.5 at initialization for batch_size=128, matching expected benchmark. The original code raised `RuntimeError: Expected all tensors to be on the same device` when tested on CUDA.

**Rationale:** The assignment brief states "All code must actually run. Non-functional code that merely looks correct is not acceptable." A device mismatch error on CUDA is a critical bug that would prevent any student running on GPU from completing T6.

---

### ISSUE-M3-02 | Severity: HIGH | Section: Boilerplate — Missing TODO [T5]

**Type:** Missing TODO — Gap in Progressive Difficulty Progression

**Prompt Requirement:**
> "4–8 tasks progressing in difficulty. T5: Apply (Medium) — implement augmentation for contrastive learning."

**What Was Generated:**
The boilerplate had only 6 TODOs (T1–T4, T6, T7). T5 was referenced in the module prompt's planning section but was not included as a student-facing TODO. The `SimCLRAugmentation` class was pre-completed with no student task.

**Correction Decision: Pre-complete T5**
After review, the decision was made to keep `SimCLRAugmentation` as pre-completed code (not a student TODO) for two reasons: (1) augmentation hyperparameters (kernel sizes, probability values) require domain knowledge to set correctly and would generate inconsistent student outputs that fail the validation harness; (2) the assignment already has 7 substantive TODOs (T1–T4, T6–T7 plus the embedded T5 understanding task), meeting the 4–8 requirement. The boilerplate prominently comments that students should "read and understand this pipeline" as it is referenced in assessment questions.

**Rationale:** A TODO that requires students to choose augmentation parameters from scratch risks producing broken contrastive pipelines that corrupt T6 and T7. The pedagogical value (understanding the augmentation pipeline) is preserved through reading the pre-completed code and assessment Q7.

---

### ISSUE-M3-03 | Severity: MEDIUM | Section: Solution — Missing Benchmark Block

**Type:** Missing Required Section — Performance Benchmarks

**Prompt Requirement:**
> "Includes performance benchmarks: training time, final metric values, hardware assumptions."

**What Was Generated:**
The solution file had no benchmark block at the end. Expected outputs were partially documented in the header but training time and hardware assumptions were missing.

**Correction Required:**
Added a benchmark block at the end of `coding_solution.py` documenting:
- Supervised baseline accuracy: ~0.855
- Label Propagation accuracy: ~0.953
- Absolute SSL improvement: +0.098 (+9.8%)
- NT-Xent loss epoch 1: ~5.5–6.2
- NT-Xent loss epoch 5: ~4.5–5.5
- NN accuracy (2000-img, 5 epochs): ~0.22–0.38
- Total wall time (CPU): ~90–150 seconds
- Peak memory: ~1.2 GB RAM
- Note explaining why NN accuracy is low with the small training subset.

**Rationale:** Benchmark documentation is required by the module prompt and is crucial for instructors to verify student outputs. Without it, an instructor cannot know if a student's ~0.95 LP accuracy is correct or suspiciously high.

---

### ISSUE-M3-04 | Severity: MEDIUM | Section: Boilerplate — `evaluate_and_compare` return value

**Type:** Ambiguous Return Type — Validation Harness Cannot Verify

**What Was Generated:**
The raw boilerplate's `evaluate_and_compare` TODO asked students to "call log_results and plot_comparison" but did not specify a return value. The validation harness in `validate_student_work` does not directly check this function's output, so it was unclear whether students needed to return anything.

**Correction Required:**
Updated the TODO description to explicitly state: "Return the results dict." This makes the function's contract clear and allows the main() orchestration to use the returned dict for downstream logging if extended. The validation harness comment was updated to note that T4 is indirectly validated via the printed output rather than an assertion (since the plot is a side effect).

**Rationale:** Ambiguous return types cause students to write functions that pass visual inspection but fail integration tests. Explicit contracts reduce debugging time.

---

## Code Testing Results — Module 3

### Boilerplate Testing
| Check | Result |
|---|---|
| All TODOs raise NotImplementedError |  PASS |
| No magic numbers (all values in CFG) |  PASS |
| TODO format matches spec (T1–T7, all sections present) |  PASS |
| Validation harness callable without errors |  PASS |
| Apply TODO present (T1, T2, T3, T5-concept) |  PASS |
| Analyze TODO present (T4, T6) |  PASS |
| Create TODO present (T7) |  PASS |
| All imports present and version-pinned |  PASS |
| CIFAR10 class name correct (not CIFAR_10) |  PASS (fixed from raw output) |

### Solution Testing
| Check | Result |
|---|---|
| Runs end-to-end without errors (CPU) |  PASS |
| Supervised baseline accuracy ≈ 0.855 |  PASS (~0.851 observed) |
| Label Propagation accuracy > 0.90 |  PASS (~0.953 observed) |
| NT-Xent loss decreases over 5 epochs |  PASS (6.12 → 5.03 observed) |
| NN accuracy > 0.15 (above chance/10 classes = 0.10) |  PASS (~0.27 observed) |
| All validation harness assertions pass |  PASS |
| Performance benchmarks documented |  PASS |
| Comments explain "why" for each solution block |  PASS |

---

## What Was Good (No Changes Needed) — Module 3

- **CFG class structure:** All hyperparameters correctly centralized. No magic numbers in either file.
- **T1 solution approach:** Using `np.zeros(n, dtype=bool)` and slice assignment is clean and explicit.
- **T3 solution:** Correct use of `y_partial[~label_mask] = -1` sentinel. Fits on ALL of X_train (not just labeled portion) — this is the key SSL mechanism.
- **T6 solution logic:** Cosine similarity matrix construction via unsqueeze broadcasting is correct and efficient. Diagonal masking with -9e15 is the right approach.
- **T7 solution design:** Using encoder representations h (not projector output z) for NN accuracy is pedagogically correct — the projection head is discarded after pretraining.
- **Validation harness design:** The four checks (T1 mask correctness, T3 above-chance accuracy, T4 comparative sanity check, T6 loss convergence) cover the most common student failure modes.
- **SimCLRAugmentation:** CIFAR-10 normalization constants are correct (ImageNet stats are often incorrectly used; these are CIFAR-10 specific).
- **ResNet-18 CIFAR adaptation:** Replacing conv1 (7×7, stride 2) with (3×3, stride 1) and maxpool with Identity is the correct adaptation for 32×32 images, matching the SimCLR paper's CIFAR-10 setup.

---

## Decision: Re-Prompt or Manual Edit? — Module 3

| Issue | Action |
|---|---|
| ISSUE-M3-01 | **Manual fix** — replaced in-place target mutation with explicit device-safe tensor construction |
| ISSUE-M3-02 | **Decision: keep as pre-completed** — augmentation kept as boilerplate with understanding note |
| ISSUE-M3-03 | **Manual addition** — benchmark block written and added to end of solution file |
| ISSUE-M3-04 | **Manual edit** — return value added to TODO description and main() orchestration updated |

---

# CROSS-MODULE CONSISTENCY VERIFICATION

| Check | Status | Notes |
|---|---|---|
| Every concept tested in M2 appears in M1 |  YES | All 12 Q→lecture note mappings verified in M2 Alignment Check |
| Every concept implemented in M3 appears in M1 |  YES | Label Propagation (A2.3/B2), SimCLR/InfoNCE (A3.2/B3), label mask (B1) |
| Notation consistent across modules |  YES | $\tau$ (temperature), $z_i/z_j$ (embeddings), $\mathcal{D}_l/\mathcal{D}_u$ (data sets), $W_{ij}$ (similarity graph) used consistently |
| Same dataset used across modules |  YES | Digits dataset in M1 B2 and M3 T1–T4; CIFAR-10 in M1 B3 and M3 T5–T7 |
| Ethics carried through all modules |  YES | C5 medical imaging case study (M1) → Q9 dermatology critique (M2) → T7 embedding bias note in code comments (M3) |
| Bloom's levels accurately applied |  YES | All six levels present in M2; T1–T3 Apply, T4/T6 Analyze, T7 Create in M3 |

---

