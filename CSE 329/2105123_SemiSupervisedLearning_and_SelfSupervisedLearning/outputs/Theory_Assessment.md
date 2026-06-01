# Theory Assessment: Semi-Supervised and Self-Supervised Learning
### CSE 329: Machine Learning | Total: 84 marks

---

## Section 1 — Remember (Recall)

---
**Q1 | Bloom's Level: Remember | Action Verb: Define | Marks: 3**

Define the three core assumptions that must hold for unlabeled data to be useful in semi-supervised learning. For each assumption, state it formally in one sentence.

**Answer Key:**
- **Model Answer:**
  1. **Smoothness Assumption:** If two points $x_1, x_2$ are close in a high-density region of input space, their labels $y_1, y_2$ should be the same.
  2. **Cluster Assumption:** Points belonging to the same cluster likely share the same label; the decision boundary should pass through low-density regions.
  3. **Manifold Assumption:** High-dimensional data lies on a lower-dimensional manifold, and unlabeled data helps map the structure of this manifold.
- **Marking Scheme:**
  - 1 mark per assumption stated correctly (name + one-sentence definition). Partial credit: 0.5 marks if the name is correct but the definition is vague (e.g., "nearby points have similar labels" without mentioning high-density regions for smoothness).
- **Expected Response Depth:** One sentence per assumption. No derivation required.

---
**Q2 | Bloom's Level: Remember | Action Verb: State | Marks: 2**

State the InfoNCE loss formula for contrastive self-supervised learning and identify what each term represents.

**Answer Key:**
- **Model Answer:**
$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$
  - $z_i, z_j$: Embeddings of the positive pair (two augmented views of the same image).
  - $z_k$: Embeddings of all other samples in the batch (negatives).
  - $\tau$: Temperature hyperparameter controlling the sharpness of the distribution.
  - $\text{sim}(\cdot, \cdot)$: Cosine similarity function.
- **Marking Scheme:** 1 mark for the correct formula structure (numerator/denominator), 1 mark for correctly identifying all four terms. Partial: 0.5 if formula is correct but two or more terms are not explained.
- **Expected Response Depth:** Formula plus a brief label for each symbol. No derivation.

---

## Section 2 — Understand (Comprehension)

---
**Q3 | Bloom's Level: Understand | Action Verb: Explain | Marks: 4**

Explain why the Generative (GMM+EM) approach to semi-supervised learning can sometimes perform *worse* than a purely supervised baseline trained on labeled data alone. Your explanation must reference the specific mechanism in the EM algorithm that causes this failure.

**Answer Key:**
- **Model Answer:** The GMM+EM approach adds a second term to the log-likelihood for unlabeled data: $\sum_{i=l+1}^{l+u} \log \sum_y p(y|\theta)p(x_i|y,\theta)$. This term is only beneficial if the assumed generative model (Gaussian class-conditionals) accurately reflects the true data distribution. If the real data is not Gaussian, or if classes overlap significantly, the E-step computes *incorrect* soft labels $\gamma_{iy}$ for unlabeled points. These erroneous soft labels then corrupt the M-step parameter updates, pushing $\mu_y$ and $\Sigma_y$ away from their optimal values. Because $u \gg l$, the unlabeled term dominates the objective, and the corrupted updates overwhelm the correct signal from labeled data — resulting in worse performance than using labeled data alone.
- **Marking Scheme:**
  - 1 mark: Identifies that the failure is due to a wrong generative assumption.
  - 1 mark: Links this to incorrect soft labels in the E-step.
  - 1 mark: Explains that corrupted soft labels propagate to M-step parameter updates.
  - 1 mark: Notes that the unlabeled term dominates because $u \gg l$.
- **Expected Response Depth:** 3–5 sentences referencing the EM equations or steps by name.

---
**Q4 | Bloom's Level: Understand | Action Verb: Compare | Marks: 4**

Compare SimCLR and BYOL along three dimensions: (a) how they prevent representational collapse, (b) their dependency on batch size, and (c) computational requirements. Explain *why* each difference exists — not just *that* it exists.

**Answer Key:**
- **Model Answer:**
  - **(a) Collapse Prevention:** SimCLR uses large batches of *negative* pairs — pushing representations of different images apart forces the model to learn discriminative features. BYOL uses a *momentum encoder* (target network) and stop-gradient: the online network must predict a slowly-evolving target that it cannot trivially copy, making constant output an unstable equilibrium.
  - **(b) Batch Size Dependency:** SimCLR requires large batches (256–4096) because each image needs many negatives within the batch to define a meaningful contrastive objective. BYOL is batch-size-agnostic because it has no negative samples — each training step involves only a positive pair.
  - **(c) Computational Cost:** SimCLR requires computing a full $(2N \times 2N)$ similarity matrix per batch — $O(N^2)$ memory. BYOL requires maintaining two networks (online + target) and computing momentum updates, but avoids the quadratic similarity matrix.
- **Marking Scheme:** 1 mark per dimension (a, b, c), split into 0.5 for *what* and 0.5 for *why*. Maximum 4 marks total (award up to 1.5 for any dimension if exceptionally well explained and another is weak).
- **Expected Response Depth:** 2–3 sentences per dimension. Must reference specific mechanisms (negative samples, momentum update equation, similarity matrix).

---

## Section 3 — Apply (Procedural Use)

---
**Q5 | Bloom's Level: Apply | Action Verb: Trace | Marks: 5**

Trace one full iteration of Label Propagation on the toy graph below. Show all intermediate calculations.

**Setup:**
- 4 nodes: nodes 1 and 2 are labeled ($y_1 = 1$, $y_2 = 0$); nodes 3 and 4 are unlabeled.
- Normalized weight matrix (row-normalized $\hat{W}$):

$$\hat{W} = \begin{pmatrix} 0 & 0.5 & 0.5 & 0 \\ 0.5 & 0 & 0 & 0.5 \\ 0.4 & 0 & 0 & 0.6 \\ 0 & 0.3 & 0.7 & 0 \end{pmatrix}$$

- Initial label matrix (one-hot, unlabeled nodes initialized to 0.5):

$$F^{(0)} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}$$

Perform the propagation step $F^{(1)} = \hat{W} F^{(0)}$, then apply the clamping step. Show $F^{(1)}$ after clamping.

**Answer Key:**
- **Model Answer:**

Step 1 — Propagation $F^{(1)} = \hat{W} F^{(0)}$:

$$F^{(1)}_{\text{raw}} = \begin{pmatrix} 0{\cdot}1 + 0.5{\cdot}0 + 0.5{\cdot}0.5 + 0{\cdot}0.5 \\ 0.5{\cdot}1 + 0{\cdot}0 + 0{\cdot}0.5 + 0.5{\cdot}0.5 \\ 0.4{\cdot}1 + 0{\cdot}0 + 0{\cdot}0.5 + 0.6{\cdot}0.5 \\ 0{\cdot}1 + 0.3{\cdot}0 + 0.7{\cdot}0.5 + 0{\cdot}0.5 \end{pmatrix} \text{ (class 1 column)} = \begin{pmatrix} 0.25 \\ 0.75 \\ 0.70 \\ 0.35 \end{pmatrix}$$

Class 0 column is the complement. So:

$$F^{(1)}_{\text{raw}} = \begin{pmatrix} 0.25 & 0.75 \\ 0.75 & 0.25 \\ 0.70 & 0.30 \\ 0.35 & 0.65 \end{pmatrix}$$

Step 2 — Clamping (restore labeled node values):

$$F^{(1)} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0.70 & 0.30 \\ 0.35 & 0.65 \end{pmatrix}$$

Node 3 would be assigned class 1; node 4 would be assigned class 0 (after argmax).

- **Marking Scheme:**
  - 2 marks for correct matrix multiplication (allow ±0.01 rounding error per entry; deduct 0.5 per wrong entry, minimum 0).
  - 1 mark for correctly identifying the clamping operation.
  - 1 mark for correctly restoring labeled node rows to ground truth.
  - 1 mark for correct final argmax class assignment for nodes 3 and 4.
- **Expected Response Depth:** Full matrix arithmetic shown. Final answer table.

---
**Q6 | Bloom's Level: Apply | Action Verb: Calculate | Marks: 4**

A SimCLR model produces the following $\ell_2$-normalized embeddings for a batch of $N=2$ images (4 vectors total, two views per image):

$$z_1 = [1, 0], \quad z_1' = [0.9, 0.436], \quad z_2 = [0, 1], \quad z_2' = [-0.2, 0.98]$$

Using temperature $\tau = 0.5$, calculate $\mathcal{L}_{1,1'}$ — the InfoNCE loss for the positive pair $(z_1, z_1')$. The denominator should sum over all vectors except $z_1$ itself.

**Answer Key:**
- **Model Answer:**

Cosine similarities from $z_1 = [1, 0]$:
- $\text{sim}(z_1, z_1') = 1 \cdot 0.9 + 0 \cdot 0.436 = 0.9$ (positive pair)
- $\text{sim}(z_1, z_2) = 1 \cdot 0 + 0 \cdot 1 = 0.0$
- $\text{sim}(z_1, z_2') = 1 \cdot (-0.2) + 0 \cdot 0.98 = -0.2$

Scaled by $\tau = 0.5$:
- Numerator: $\exp(0.9 / 0.5) = \exp(1.8) \approx 6.050$
- Denominator: $\exp(1.8) + \exp(0.0/0.5) + \exp(-0.2/0.5) = 6.050 + 1.000 + \exp(-0.4) \approx 6.050 + 1.000 + 0.670 = 7.720$

$$\mathcal{L}_{1,1'} = -\log\frac{6.050}{7.720} \approx -\log(0.7836) \approx 0.244$$

- **Marking Scheme:**
  - 1 mark: Correct cosine similarities computed (all three).
  - 1 mark: Correct exponentials computed.
  - 1 mark: Correct denominator (three terms, not four — $z_1$ is excluded).
  - 1 mark: Final loss value within ±0.02.
- **Expected Response Depth:** Step-by-step arithmetic shown. Do not accept a final number without work.

---

## Section 4 — Analyze (Decomposition)

---
**Q7 | Bloom's Level: Analyze | Action Verb: Diagnose | Marks: 6**

A research team applies SimCLR to a dataset of satellite images to learn representations for land-cover classification. After 100 epochs of pretraining, linear probe accuracy is only 28% — barely above the random baseline of 25% for a 4-class problem. The team reports that training loss decreased steadily from 6.1 to 5.9 across 100 epochs.

Identify **two specific failure mechanisms** that could explain this result. For each, explain (a) what is happening mathematically or architecturally, and (b) what evidence in the scenario supports your diagnosis.

**Answer Key:**
- **Model Answer (two of the following three are acceptable):**

  **Failure 1: Augmentation Mismatch** — SimCLR's representations are only as useful as the augmentations are semantically appropriate. Satellite images differ fundamentally from natural images: color jitter, which is invariant for object recognition, *destroys* semantically important information in satellite imagery (e.g., vegetation indices depend on color). If the team used standard CIFAR/ImageNet augmentations, the model learned invariances that conflict with the features needed for land-cover classification. Evidence: the loss *did* decrease (the model learned *some* invariances), but the features are wrong for the task.

  **Failure 2: Mode Collapse (Near-Collapse)** — The loss decreasing only from 6.1 to 5.9 over 100 epochs is a very small absolute decrease. For a healthy SimCLR run, the loss should drop significantly in early epochs. This near-flat curve suggests the model's representations have partially collapsed — all embeddings are living in a low-dimensional subspace, so the contrastive task is "solved" in a degenerate way. This can happen with too-small batch sizes (insufficient negatives) or too-high temperature. Evidence: minimal loss improvement despite 100 epochs.

  **Failure 3: Domain-Inappropriate Backbone** — ResNet-18 pretrained architecture assumes $224 \times 224$ natural images. Satellite images may have different spatial statistics, different spectral bands, or very different scales of objects. If patches from random crops do not contain semantically consistent regions (e.g., a crop of 10 pixels might straddle two land-cover classes), positive pairs are not truly "views of the same semantic concept." Evidence: the task involves 4-class classification but accuracy is near chance — the model is not capturing the relevant features.

- **Marking Scheme:**
  - 3 marks per failure: 1.5 for mechanism (a), 1.5 for evidence link (b). Award partial credit if mechanism is correct but evidence link is weak (1 mark).
  - Full credit requires reference to either the mathematical objective, batch size, or augmentation strategy.
- **Expected Response Depth:** 4–6 sentences per failure. Novel scenario — not a lecture note example.

---
**Q8 | Bloom's Level: Analyze | Action Verb: Differentiate | Marks: 5**

A practitioner has a dataset of 10,000 images: 200 are labeled, 9,800 are unlabeled. They must choose between Label Propagation and Self-Training. Analyze the trade-offs for this *specific* dataset scale. Identify which method is more appropriate and explain **three concrete factors** that drive your recommendation. Do not give a generic "it depends" answer.

**Answer Key:**
- **Model Answer:** Self-Training is likely more appropriate at this scale for the following three reasons:

  1. **Scalability of graph construction:** Label Propagation requires constructing a pairwise similarity matrix $W \in \mathbb{R}^{10000 \times 10000}$ — storing this requires $\sim$800 MB for float64, and eigendecomposition or iterative propagation on this graph is expensive. Self-training operates on individual predictions and does not require the full graph.

  2. **Flexibility of the base classifier:** Self-training can use any classifier (neural network, gradient boosted trees) that produces confidence scores. Label propagation is tightly coupled to the RBF kernel and the graph structure — its performance degrades when the manifold assumption is weak (e.g., images of 10,000 diverse scenes may not form clean clusters in raw pixel space).

  3. **Iterative label quality control:** Self-training uses a confidence threshold $\delta$ that allows the practitioner to inspect and adjust which unlabeled samples get absorbed into the labeled set. Label propagation commits to all labels simultaneously in a single closed-form solution, offering no intermediate checkpoint to catch errors.

  *However*, if the image embeddings form clean clusters (verifiable via t-SNE), Label Propagation on a pre-extracted feature space (not raw pixels) could outperform Self-Training. This caveat should be noted.

- **Marking Scheme:**
  - 2 marks for identifying Self-Training as more appropriate (or providing a well-justified alternative recommendation).
  - 1 mark per concrete factor (3 × 1 = 3 marks). Must reference specific mechanisms. Generic statements ("Label Propagation is slower") earn 0 — student must explain *why* at this scale.
- **Expected Response Depth:** One paragraph per factor. Total ~200 words.

---

## Section 5 — Evaluate (Judgment)

---
**Q9 | Bloom's Level: Evaluate | Action Verb: Critique | Marks: 7**

A startup is building a medical diagnosis system for skin lesion classification. Their engineer proposes to use SimCLR pretraining on 500,000 dermatology images scraped from public forums (Reddit, patient blogs), followed by fine-tuning on 800 labeled images from a single dermatology clinic.

Critically evaluate this proposal. Your evaluation must address: (a) what could go wrong with the pretraining data source specifically, (b) whether SimCLR's contrastive objective is well-suited to this domain, and (c) what responsible alternative you would recommend and why.

**Answer Key — Sample High-Scoring Response:**

**(a) Pretraining data source:** Public forum images are not clinically curated. They suffer from three specific SSL risks: (1) *Representation bias* — forum users posting skin lesion photos skew toward certain demographics (lighter skin tones, English-speaking users), meaning the self-supervised encoder will learn features optimized for this subgroup. Darker skin tones, which are underrepresented in most dermatology datasets, will have poor feature coverage. (2) *Label noise in pretraining* — while SSL does not use explicit labels, images from Reddit may include mislabeled or misidentified conditions in captions that could influence learned representations if text is used as a secondary signal. (3) *Privacy violation* — patients posting images of medical conditions may not have consented to their images being used to train commercial medical AI systems, constituting a potential privacy and ethical violation even if the data is technically "public."

**(b) SimCLR suitability:** SimCLR's augmentation strategy assumes that color-jittered or cropped views of the same image are semantically equivalent. In dermatology, this is *partially* true — the same lesion photographed with different lighting should map to the same diagnosis. However, SimCLR's augmentations include random cropping and color jitter that can remove diagnostic features (e.g., dermoscopic color patterns — specific hues are diagnostically significant in ABCD criteria for melanoma). The model may learn illumination-invariant features while discarding color information that a clinician would use for diagnosis.

**(c) Responsible alternative:** Use a pretrained medical foundation model (e.g., pretrained on ISIC dataset or similar curated dermatology datasets) rather than web-scraped data, combined with domain-adaptive self-supervised pretraining on a small set of clinically consented images from multiple demographic groups. Fine-tuning should include data from multiple clinics to reduce single-site bias. Before deployment, perform subgroup analysis disaggregated by Fitzpatrick skin type.

**Why this earns full marks:** It references (i) a specific SSL mechanism (augmentation invariance in contrastive learning), (ii) a domain-specific failure mode (color as diagnostic feature), and (iii) an ethical issue tied to the technique (representation bias from uncurated web data), not generic AI ethics boilerplate.

- **Marking Scheme:**
  - (a) 2.5 marks: 1 for identifying representation bias, 1 for privacy concern, 0.5 for connecting to SSL-specific mechanism.
  - (b) 2 marks: 1 for identifying augmentation-domain mismatch, 1 for the specific diagnostic color example.
  - (c) 2.5 marks: 1 for a concrete responsible alternative, 1 for justifying it with reference to SSL properties, 0.5 for mentioning evaluation disaggregation.
- **Expected Response Depth:** ~300 words. Full marks require all three parts and explicit SSL mechanism references.

---
**Q10 | Bloom's Level: Evaluate | Action Verb: Justify | Marks: 6**

A machine learning engineer argues: *"Self-training is strictly better than label propagation because it can use any classifier, while label propagation is limited to kernel methods."*

Assess the validity of this argument. Identify one condition under which it is correct, one condition under which it is incorrect, and justify your positions with reference to specific algorithmic properties.

**Answer Key:**
- **Model Answer:**

  The argument is *partially* valid but overstated.

  **When correct:** If the data distribution is non-Gaussian and does not form clean clusters in feature space (e.g., a high-dimensional embedding space where manifold geometry is complex), the kernel-based RBF graph in label propagation will produce incorrect edges, spreading labels across class boundaries. A neural network-based self-training classifier can learn non-linear decision boundaries that are more appropriate. The argument is valid in this setting.

  **When incorrect:** Label propagation can be applied to *any* kernel-computed similarity, not just RBF. Moreover, label propagation uses a *global* closed-form solution that considers all unlabeled points simultaneously, while self-training greedily adds the most confident predictions — suffering from confirmation bias (errors compound over iterations). On datasets where the manifold assumption holds (e.g., image datasets with well-separated semantic clusters), label propagation consistently outperforms self-training at the same label budget. Furthermore, self-training's performance depends critically on the initial base classifier, which is itself trained on very few labels — if the initial classifier is poor, the pseudo-labels will be wrong from the start.

- **Marking Scheme:**
  - 2 marks for correctly identifying the condition where the argument is correct (with justification).
  - 2 marks for correctly identifying a condition where it is incorrect (with justification).
  - 2 marks for quality of reasoning — must reference specific algorithmic properties (kernel vs. non-linear boundary, confirmation bias, manifold assumption).
- **Expected Response Depth:** ~150 words. Balanced argument with specific references.

---

## Section 6 — Create (Synthesis)

---
**Q11 | Bloom's Level: Create | Action Verb: Design | Marks: 10**

You are building a wildlife monitoring system for a conservation organization. You have 5,000 camera trap images with no labels and 100 labeled images identifying 8 animal species. The organization wants to automatically classify new images in real time on a solar-powered edge device with 2 GB RAM and no internet connection.

**Design a complete self-supervised + semi-supervised learning pipeline** for this system. Your design must specify:
1. Which self-supervised pretraining method to use and why (with reference to data volume and compute constraints).
2. How to construct and apply the semi-supervised fine-tuning step using the 100 labeled images.
3. How you will evaluate whether the pipeline is actually working before deployment.
4. One specific failure mode tied to this deployment context and how you will mitigate it.

**Answer Key — Rubric for Open-Ended Question:**

**(1) Pretraining Method (3 marks):**
An excellent answer chooses MAE or a MobileNet-based SimCLR (not standard ResNet-50) and justifies the choice with respect to:
- *Data volume:* 5,000 images is small for SimCLR (needs large batches); MAE is effective at smaller scales.
- *Edge device constraint:* MAE's encoder can be a ViT-Tiny (~5M parameters), which fits in 2 GB RAM. ResNet-18 is also acceptable if justified.
- *No internet:* Cannot use pretrained weights — must train from scratch. This makes MAE's reconstruction objective more stable than contrastive learning with small batches.

**(2) Semi-supervised fine-tuning (3 marks):**
An excellent answer specifies:
- Use Label Propagation (or self-training) on the 5,000-image feature space extracted from the pretrained encoder.
- 100 labeled images → train with label propagation on nearest-neighbor graph in the embedding space.
- Alternatively: self-training with a KNN classifier base (more memory-efficient than a neural classifier on 2 GB edge device).

**(3) Evaluation (2 marks):**
- Held-out test set from the 100 labeled images (stratified split, e.g., 80/20).
- Confusion matrix disaggregated by species — rare species accuracy matters as much as overall accuracy.
- Embedding space visualization (t-SNE) to confirm species form clean clusters before deployment.

**(4) Failure mode + mitigation (2 marks):**
Excellent answers name *distribution shift* (new species appear after deployment, or seasonal appearance changes confound the model) or *class imbalance* (rare species underrepresented in 5,000 unlabeled images, so the encoder learns poor features for them). Mitigation: track prediction confidence distribution over time; flag consistently uncertain predictions for human review; periodically fine-tune on newly labeled data.

- **Marking Scheme:** Rubric-based. Award marks per sub-question as above. Partial credit: if student addresses 3 of 4 sub-questions well, award 7/10.
- **Expected Response Depth:** ~400 words. Specific methods named and justified. Generic answers ("use deep learning") earn 0 for that sub-question.

---
**Q12 | Bloom's Level: Create | Action Verb: Propose | Marks: 8**

The co-training algorithm (Blum & Mitchell, 1998) requires two *conditionally independent* views of the data. Propose a co-training-inspired system for a dataset of electronic health records (EHR) where each patient has: (a) structured data (lab values, vitals) and (b) unstructured clinical notes (free text). Your system should classify patients as high-risk or low-risk for hospital readmission.

Specify: how you will define the two views, what classifier you will use for each view, how you will verify the independence assumption, and what you will do if the assumption is violated.

**Answer Key — Rubric:**

**(View definition, 2 marks):**
- View 1: Structured EHR features — tabular features (serum creatinine, heart rate, medication list). Use a gradient-boosted tree (e.g., XGBoost) — well-suited to tabular data, produces calibrated probability scores.
- View 2: Clinical notes — free-text features processed with a pretrained language model (e.g., BioBERT or ClinicalBERT fine-tuned on MIMIC notes). Produces embedding-based predictions.

**(Independence verification, 2 marks):**
- Compute mutual information between the prediction vectors from both classifiers on a held-out subset. Low MI suggests approximate independence.
- Check correlation between features: lab values and clinical note mention of the same lab test are correlated (e.g., "creatinine 3.2 mg/dL" appears in notes). This violates independence. Student should identify this specific risk.

**(Independence violation handling, 2 marks):**
- If independence is violated: Remove redundant features from one view (e.g., don't include lab values in the NLP features — use a note embedding model that ignores numerical lab values).
- Fall back to a different algorithm: if views are too correlated, self-training with a multimodal classifier is more appropriate than co-training.

**(Overall system quality, 2 marks):**
- Is the system feasible? Does the student address the 8-to-2 label ratio (co-training is beneficial precisely when labels are scarce)?
- Does the student account for clinical constraints (interpretability, HIPAA compliance for model training data)?

- **Marking Scheme:** 2 marks per sub-question. Partial credit for partially correct reasoning.
- **Expected Response Depth:** ~300 words. Must name specific classifiers, not "some model."

---

## Assessment Summary Table

| Q# | Section | Bloom's Level | Action Verb | Topic Area | Marks |
|----|---------|---------------|-------------|------------|-------|
| Q1 | 1 | Remember | Define | SSL Assumptions | 3 |
| Q2 | 1 | Remember | State | InfoNCE Loss Formula | 2 |
| Q3 | 2 | Understand | Explain | GMM+EM Failure Mode | 4 |
| Q4 | 2 | Understand | Compare | SimCLR vs BYOL | 4 |
| Q5 | 3 | Apply | Trace | Label Propagation Algorithm | 5 |
| Q6 | 3 | Apply | Calculate | InfoNCE Loss Computation | 4 |
| Q7 | 4 | Analyze | Diagnose | SimCLR Failure Analysis | 6 |
| Q8 | 4 | Analyze | Differentiate | LP vs Self-Training Trade-offs | 5 |
| Q9 | 5 | Evaluate | Critique | Medical SSL Ethical Design | 7 |
| Q10 | 5 | Evaluate | Justify | Self-Training vs LP Argument | 6 |
| Q11 | 6 | Create | Design | End-to-End SSL Pipeline | 10 |
| Q12 | 6 | Create | Propose | Co-Training EHR System | 8 |
| **Total** | | | | | **64** |

---

## Alignment Check

```
ALIGNMENT CHECK:
- Every question maps to a concept in the lecture note: YES
  Q1→A2.1, Q2→A3.2, Q3→A2.2, Q4→A3.3, Q5→A2.3, Q6→A3.2,
  Q7→A3.2 (novel scenario), Q8→A2.3/A2.4 (novel scenario),
  Q9→C2/C3 (novel scenario), Q10→A2.3/A2.4 (novel scenario),
  Q11→A3/B1 (novel scenario), Q12→A2.4 (novel scenario)
- Bloom's levels correctly applied (no recall questions mislabeled as analysis): YES
  Q1-Q2 are pure recall (define/state formula). Q3-Q4 require explanation of mechanism (not recall).
  Q5-Q6 require computation/trace. Q7-Q8 present novel scenarios. Q9-Q10 require judgment with evidence. Q11-Q12 require synthesis.
- At least one Evaluate/Create question addresses ethics: YES — Q9 (Evaluate) addresses
  representation bias, privacy risk of web-scraped pretraining data, and domain-specific
  augmentation failure. Q11 (Create) requires identifying a deployment failure mode.
- Novel scenarios used at levels 4–6: YES
  Q7: Satellite imagery (not in lecture note). Q8: 10,000-image scale trade-off analysis (not in lecture note).
  Q9: Skin lesion forum data (not in lecture note). Q10: Evaluating a specific engineer's argument (not in lecture note).
  Q11: Wildlife edge device deployment (not in lecture note). Q12: EHR co-training (not in lecture note).
- All answer keys include partial credit rules: YES — all marking schemes specify partial credit conditions.
```
