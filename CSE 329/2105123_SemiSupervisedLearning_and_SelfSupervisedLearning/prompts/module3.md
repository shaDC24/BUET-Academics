# Module 3 Prompt — Coding Assessment Generator


---

## Context

You will generate two complete Python files for a coding assessment on **Semi-Supervised and Self-Supervised Learning**:

1. `coding_boilerplate.py` — student-facing file with TODO stubs
2. `coding_solution.py` — complete instructor solution

The task: **implement a semi-supervised classification pipeline using Label Propagation and compare it against a supervised baseline, then implement the NT-Xent (InfoNCE) contrastive loss for self-supervised learning.**

This task was chosen because it directly implements the two most important concepts from the lecture note, covers the full spectrum from Apply to Create in Bloom's taxonomy, and is achievable in a single runnable script.

---

## Step 1 — Chain-of-Thought Planning (Internal, Do Not Print)

Before writing code, think through:

1. What is the minimal complete pipeline that covers semi-supervised AND self-supervised concepts?
   - Semi-supervised: Label Propagation on a partially labeled dataset with evaluation.
   - Self-supervised: NT-Xent loss implementation and contrastive pair construction.
2. What should be pre-completed (boilerplate) vs left for the student?
   - Pre-complete: imports, config, data loading, plotting, logging, main().
   - Student implements: label mask construction, label propagation call, accuracy evaluation, NT-Xent loss, contrastive pair construction, comparative analysis.
3. How do we ensure progressive difficulty? Map TODO difficulty to Bloom's levels:
   - T1: Apply (Easy) — create the label mask
   - T2: Apply (Easy) — train supervised baseline
   - T3: Apply (Medium) — run Label Propagation
   - T4: Analyze (Medium) — evaluate and compare methods
   - T5: Apply (Medium) — implement augmentation for contrastive learning
   - T6: Analyze (Hard) — implement NT-Xent loss
   - T7: Create (Hard) — design a validation function for self-supervised embeddings
4. What validation harness checks are meaningful?
   - Accuracy above random chance (>50% on binary task)
   - Label mask has correct number of labeled samples
   - NT-Xent loss decreases over first 5 iterations
   - Embeddings are L2-normalized

Use this plan to write the code. Do not print the plan.

---

## Step 2 — Few-Shot Example of TODO Format

Every TODO block must follow **exactly** this template (no exceptions):

```python
# ─────────────────────────────────────────────────────────────
# TODO [T1]: Create the label mask
# Bloom's Level: Apply
# Difficulty: Easy | Expected lines: ~5
# Description:
#   Create a boolean mask `label_mask` of shape (n_samples,) where
#   True indicates a labeled sample. Use CFG.labeled_fraction to
#   determine how many samples are labeled. Labeled samples should
#   be drawn from the BEGINNING of the dataset (first l samples).
#   This mask will be used to hide labels from the propagation algorithm.
# Hints:
#   1. (Conceptual) In SSL, we simulate having few labels by treating
#      most of the training set as unlabeled — the mask encodes this.
#   2. (Implementation) np.zeros(n, dtype=bool) creates a boolean array;
#      set the first int(n * CFG.labeled_fraction) entries to True.
# Expected behavior: label_mask.sum() == int(n_train * CFG.labeled_fraction)
# ─────────────────────────────────────────────────────────────
# >>> YOUR CODE HERE <<<

raise NotImplementedError("TODO T1 not yet implemented")

# >>> END YOUR CODE <<<
```

Use exactly this format (dashes, labels, sections) for every TODO.

---

## Step 3 — Generate the Boilerplate File

Generate `coding_boilerplate.py` with the following structure, in this exact order:

### Header Block
```python
"""
coding_boilerplate.py
=====================
CSE 329: Machine Learning — Coding Assessment
Topic: Semi-Supervised and Self-Supervised Learning

Python version: 3.10+
Required libraries:
    numpy==1.24.x
    scikit-learn==1.3.x
    torch==2.1.x
    torchvision==0.16.x
    matplotlib==3.7.x

Dataset: scikit-learn Digits dataset (built-in, no download required)
         CIFAR-10 (downloaded automatically via torchvision)

Run: python coding_boilerplate.py
"""
```

### Imports Block (pre-completed, with version checks)
Include all necessary imports. After imports, add a version check block:
```python
# Version check
import sklearn; assert sklearn.__version__.startswith("1."), ...
import torch; assert int(torch.__version__.split(".")[0]) >= 2, ...
```

### Config Block (pre-completed, no magic numbers)
```python
class CFG:
    # Data
    labeled_fraction: float = 0.10   # fraction of training data that is labeled
    test_size: float = 0.20
    random_seed: int = 42

    # Label Propagation
    lp_kernel: str = "knn"
    lp_n_neighbors: int = 7
    lp_max_iter: int = 1000

    # Contrastive Learning
    batch_size: int = 128
    temperature: float = 0.07
    embedding_dim: int = 128
    contrastive_epochs: int = 5        # kept low for assignment runtime

    # Paths
    output_dir: str = "./outputs"
```

### Data Loading Block (pre-completed)
Complete code to load the Digits dataset, split into train/test, and normalize.

### Logging and Plotting Utilities (pre-completed)
A `plot_comparison()` function that takes a dict of method names to accuracy scores and produces a bar chart saved to `CFG.output_dir`.
A `log_results()` function that prints a formatted table.

### TODO Sections (T1 through T7)
Insert all 7 TODOs using the exact format from Step 2. Each TODO must be preceded by a clear section comment like:
```python
# ════════════════════════════════════════════════════════════
# SECTION 2: Semi-Supervised Learning — Label Propagation
# ════════════════════════════════════════════════════════════
```

### Validation Harness (pre-completed)
```python
def validate_student_work(label_mask, lp_accuracy, baseline_accuracy, loss_values):
    """
    Call this function to check your implementations before submitting.
    All assertions should pass if your implementations are correct.
    """
    # Check T1
    assert label_mask.dtype == bool, "label_mask must be boolean"
    expected_labeled = int(len(label_mask) * CFG.labeled_fraction)
    assert label_mask.sum() == expected_labeled, \
        f"Expected {expected_labeled} labeled samples, got {label_mask.sum()}"
    print("✓ T1: label_mask is correct")

    # Check T3
    assert lp_accuracy > 0.5, \
        f"Label Propagation accuracy {lp_accuracy:.3f} is below random chance"
    print(f"✓ T3: Label Propagation accuracy = {lp_accuracy:.3f}")

    # Check T4
    assert lp_accuracy > baseline_accuracy - 0.05, \
        "SSL should not perform drastically worse than supervised baseline"
    print("✓ T4: Comparative evaluation passed")

    # Check T6
    assert len(loss_values) > 0, "No loss values recorded"
    assert loss_values[-1] < loss_values[0] * 1.5, \
        "NT-Xent loss is not converging (loss increased by >50%)"
    print(f"✓ T6: NT-Xent loss initial={loss_values[0]:.4f}, final={loss_values[-1]:.4f}")

    print("\n✅ All validation checks passed!")
```

### Main Entry Point (pre-completed skeleton)
```python
def main():
    print("=" * 60)
    print("CSE 329 | Semi-Supervised & Self-Supervised Learning")
    print("=" * 60)
    set_seed(CFG.random_seed)
    X_train, X_test, y_train, y_test = load_data()
    # Students' implementations are called from here
    # [pre-written orchestration code that calls student functions]

if __name__ == "__main__":
    main()
```

---

## Step 4 — Generate the Solution File

Generate `coding_solution.py` with:

### Header Block
```python
"""
coding_solution.py
==================
INSTRUCTOR SOLUTION — Do not distribute to students.

Expected outputs (on a standard laptop, CPU-only):
    Supervised baseline accuracy:     ~0.85–0.92
    Label Propagation accuracy:       ~0.92–0.97
    NT-Xent loss (epoch 1):           ~5.0–6.0
    NT-Xent loss (epoch 5):           ~4.2–5.5
    Total runtime:                    ~60–120 seconds

Hardware assumptions: CPU-only, 8 GB RAM, no GPU required.
"""
```

For every TODO section, provide:
```python
# ── SOLUTION T1 ──────────────────────────────────────────────
# Approach: We create a boolean mask where the first l=int(n*frac)
# entries are True (labeled). The rest simulate unlabeled data.
# LabelPropagation in sklearn uses -1 as the "unlabeled" sentinel,
# so we apply this mask when constructing y_train_partial.

label_mask = np.zeros(len(X_train), dtype=bool)
label_mask[:int(len(X_train) * CFG.labeled_fraction)] = True
# ─────────────────────────────────────────────────────────────
```

### End-of-file Benchmark Block
```python
# ════════════════════════════════════════════════════════════
# PERFORMANCE BENCHMARKS (verified on: MacBook Pro M2, CPU-only)
# ════════════════════════════════════════════════════════════
# Supervised baseline (LogisticRegression, 10% labels): 0.872
# Label Propagation (knn, k=7, 10% labels):             0.953
# Improvement from unlabeled data:                      +8.1%
# NT-Xent initial loss:                                 5.41
# NT-Xent final loss (5 epochs):                        4.87
# Total wall time:                                      ~95 seconds
```

---

## Constraints

- Both files must be complete and runnable — no ellipsis (`...`) or `pass` in the solution.
- No magic numbers anywhere — all hyperparameters must reference `CFG`.
- Every function must have a docstring.
- The boilerplate must run without errors (all TODOs raise `NotImplementedError`).
- The solution must run end-to-end and produce numerical output matching the benchmark block.
- Library versions must be explicit in both file headers.
- The validation harness in the boilerplate must be callable by the student at any point.

---

## Step 5 — Post-Generation Checklist

After generating both files, output this block:

```
CODE CHECKLIST:
Boilerplate:
  [ ] Runs without errors (all TODOs raise NotImplementedError)
  [ ] No magic numbers (all values in CFG)
  [ ] TODO format matches spec exactly (T1–T7, all sections present)
  [ ] Validation harness included and callable
  [ ] At least one TODO at Apply, Analyze, and Create level
  [ ] All imports present and version-pinned

Solution:
  [ ] Every TODO filled with correct code
  [ ] Solution runs end-to-end without errors
  [ ] Expected outputs documented in header
  [ ] Performance benchmarks included
  [ ] Comments explain the "why" behind each approach
  [ ] All validation harness checks pass
```
