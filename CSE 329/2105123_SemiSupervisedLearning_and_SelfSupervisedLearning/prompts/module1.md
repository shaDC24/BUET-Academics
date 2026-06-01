# Module 1 Prompt — Lecture Note Generator


---

## Instructions to LLM

You will now generate a comprehensive lecture note for the topic:

> **TOPIC = "Semi-Supervised and Self-Supervised Learning"**

This note is for upper-undergraduate ML students in CSE 329. Follow the chain-of-thought process below **before** writing any content, then produce the three required components in order.

---

## Step 1 — Chain-of-Thought Planning 

Before writing, think through the following questions step by step:

1. What are the core sub-topics under "Semi-Supervised and Self-Supervised Learning"?
   - Think: What is the motivation? What methods exist? How do they differ?
2. What mathematical foundations does a student need?
   - Think: probability, generative models, information theory, contrastive objectives.
3. What Python libraries and datasets are most appropriate for implementation?
   - Think: scikit-learn, PyTorch, torchvision, HuggingFace.
4. What are the most common student misconceptions about this topic?
   - Think: confusing semi-supervised with self-supervised, misunderstanding why unlabeled data helps.
5. What are the most concrete ethical risks tied specifically to this technique?
   - Think: label propagation amplifying existing biases, self-supervised models learning harmful representations from uncurated web data.

Use these answers to structure the note. Do not print this planning step — go directly to the content.

---

## Step 2 — Generate the Lecture Note

Produce the lecture note in exactly this structure. Do not skip or reorder sections.

---

# Lecture Note: Semi-Supervised and Self-Supervised Learning
### CSE 329: Machine Learning

---

## Part A — Knowledge Component

### A1. Motivation and Problem Setting
- Explain the labeled data bottleneck with a concrete real-world example.
- Define formally:
  - Supervised Learning: training set $\{(x_i, y_i)\}_{i=1}^{n}$
  - Semi-Supervised Learning (SSL): labeled set $\{(x_i, y_i)\}_{i=1}^{l}$ plus unlabeled set $\{x_i\}_{i=l+1}^{l+u}$ where $u \gg l$
  - Self-Supervised Learning (Self-SL): no labels at all; supervision signal derived from the data itself
- Provide a comparison table: Supervised vs Semi-Supervised vs Self-Supervised vs Unsupervised.

### A2. Theoretical Foundations of Semi-Supervised Learning
Cover each of the following with full mathematical derivations:

**A2.1 Key Assumptions**
- Smoothness assumption
- Cluster assumption
- Manifold assumption
Explain each assumption, when it holds, and when it breaks.

**A2.2 Generative Approach — Gaussian Mixture Models (GMM) with EM**
- Define the GMM: $p(x, y | \theta) = p(y|\theta) \cdot p(x|y, \theta) = w_y \mathcal{N}(x; \mu_y, \Sigma_y)$
- Derive the log-likelihood for the semi-supervised case:
$$\log p(X_l, Y_l, X_u | \theta) = \sum_{i=1}^{l} \log p(y_i|\theta)p(x_i|y_i,\theta) + \sum_{i=l+1}^{l+u} \log \sum_{y} p(y|\theta)p(x_i|y,\theta)$$
- Describe the EM algorithm step by step (E-step and M-step) with pseudocode.
- Discuss the risk: when the generative assumption is wrong, unlabeled data can hurt performance.

**A2.3 Graph-Based Methods — Label Propagation**
- Construct the similarity graph: $W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$
- Define the label propagation update rule.
- Explain the convergence guarantee.

**A2.4 Self-Training and Co-Training**
- Self-training pseudocode and analysis.
- Co-training (Blum & Mitchell, 1998): two-view assumption, algorithm, and when it works.

### A3. Theoretical Foundations of Self-Supervised Learning
Cover each of the following:

**A3.1 Core Idea: Pretext Tasks**
- Define a pretext task with 3 concrete examples (masked prediction, rotation prediction, jigsaw puzzles).

**A3.2 Contrastive Learning**
- Define the contrastive objective (InfoNCE loss):
$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}\right]$$
- Explain each term: anchor, positive, negatives, temperature $\tau$.
- Describe SimCLR: augmentation strategy, projection head, training procedure.

**A3.3 Non-Contrastive Methods**
- BYOL: explain how it avoids collapse without negatives (momentum encoder, stop-gradient).
- Masked Autoencoders (MAE): masking ratio, reconstruction objective, encoder-decoder design.

**A3.4 Comparison Table**
Produce a table comparing: GMM+EM, Label Propagation, Self-Training, Co-Training, SimCLR, BYOL, MAE across: type (semi/self), requires labels, scalability, typical domain, key assumption.

---

## Part B — Skill Component

### B1. Implementation Roadmap
Present a clear end-to-end workflow as a numbered list:
1. Data preparation (labeled + unlabeled split strategy)
2. Baseline supervised model
3. Semi-supervised or self-supervised method
4. Fine-tuning / downstream evaluation
5. Comparison and reporting

### B2. Code Example 1 — Semi-Supervised Learning with Label Propagation (scikit-learn)

**Few-Shot Example Format** (follow this exact structure for every code block):

```python
# ============================================================
# WHAT: Label Propagation for semi-supervised classification
# WHY: Demonstrates how unlabeled data improves decision boundary
# DATASET: Digits dataset (scikit-learn), 10% labeled
# EXPECTED OUTPUT: Accuracy improvement over supervised baseline
# ============================================================
# [full, runnable code here]
```

Produce complete, runnable code that:
- Loads the digits dataset
- Creates a realistic labeled/unlabeled split (10% labeled)
- Trains a supervised baseline (LogisticRegression)
- Trains LabelPropagation on the full set
- Reports and compares accuracy with a printed table

### B3. Code Example 2 — Self-Supervised Pretraining with SimCLR (PyTorch)

Follow the same few-shot format. Produce complete code that:
- Defines the augmentation pipeline (RandomCrop, ColorJitter, GaussianBlur)
- Defines the encoder (ResNet-18 backbone) + projection head (2-layer MLP)
- Implements the NT-Xent (InfoNCE) loss
- Runs a short pretraining loop on CIFAR-10
- Evaluates with linear probe (freeze encoder, train linear classifier)

### B4. Common Pitfalls and Debugging Guide
List at least 6 pitfalls in this format:

| # | Pitfall | Symptom | Fix |
|---|---------|---------|-----|
| 1 | [pitfall] | [what the student sees] | [concrete fix] |

### B5. Hyperparameter Tuning Guide
Provide a table:

| Hyperparameter | Method | Recommended Range | Search Strategy | Notes |
|---|---|---|---|---|
| Temperature $\tau$ | SimCLR | 0.07–0.5 | Grid search | ... |
| [continue for 6+ hyperparameters] | | | | |

---

## Part C — Ethical Implications & Values Component

### C1. Bias Amplification in Semi-Supervised Learning
- Explain specifically how label propagation can amplify label bias: if labeled examples are from a non-representative subset, the propagated labels will reflect that bias across the unlabeled data.
- Give a concrete example: medical imaging where labeled data is from one hospital demographic.

### C2. Privacy Risks
- Explain membership inference risk: self-supervised models trained on web data may memorize and expose private images.
- Discuss federated semi-supervised learning as a mitigation.

### C3. Societal Impact
- Who benefits from SSL/Self-SL? (Low-resource communities that cannot afford large labeling campaigns)
- Who may be harmed? (Groups underrepresented in unlabeled web-scale data)
- Under what conditions does self-supervised pretraining perpetuate or reduce inequality?

### C4. Responsible Practices Checklist
Produce a concrete checklist (minimum 6 items) a practitioner should follow when using these techniques.

### C5. Ethical Case Study
Write a detailed case study (minimum 300 words) on the following scenario:

> A hospital system uses a self-supervised model pretrained on 10 million unlabeled medical images scraped from public databases to build a diagnostic classifier. The pretrained model is fine-tuned on 500 labeled images from a single urban hospital and deployed across rural clinics.

Address: what could go wrong, why it could go wrong (linked to specific SSL/Self-SL mechanisms), how it could be detected, and what responsible alternatives exist.

---

## Step 3 — Final Consistency Check

After generating all three components, output this verification block:

```
CONSISTENCY CHECK:
- Core concepts covered in Part A: [list them]
- Concepts testable in Module 2 (Assessment): [list 6 concept areas]
- Concepts implementable in Module 3 (Code): [list 4 implementation tasks]
- Ethical themes carried forward: [list 2]
- Notation used consistently: [list key symbols and their meanings]
```
