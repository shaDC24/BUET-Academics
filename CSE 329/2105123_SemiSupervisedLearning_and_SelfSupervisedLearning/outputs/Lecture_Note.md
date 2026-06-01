# Lecture Note: Semi-Supervised and Self-Supervised Learning
### CSE 329: Machine Learning

---

## Part A — Knowledge Component

### A1. Motivation and Problem Setting

In modern machine learning, we often face a "data paradox": we have access to massive amounts of raw data (images, text, sensor logs), but labeling that data is expensive, time-consuming, and requires domain expertise. For example, in **medical imaging**, a hospital may have 100,000 X-ray scans, but only 500 have been annotated by a specialized radiologist.

**Formal Definitions:**
*   **Supervised Learning:** Training on a labeled dataset $\mathcal{D}_l = \{(x_i, y_i)\}_{i=1}^{n}$.
*   **Semi-Supervised Learning (SSL):** Training on a small labeled set $\mathcal{D}_l = \{(x_i, y_i)\}_{i=1}^{l}$ and a large unlabeled set $\mathcal{D}_u = \{x_i\}_{i=l+1}^{l+u}$, where $u \gg l$.
*   **Self-Supervised Learning (Self-SL):** Training on $\mathcal{D}_u$ by creating "pseudo-labels" from the data itself (e.g., predicting the next word or a rotated image) to learn a useful representation $z = f(x)$.

| Feature | Supervised | Semi-Supervised | Self-Supervised | Unsupervised |
| :--- | :--- | :--- | :--- | :--- |
| **Labels Required** | Full ($y$ for every $x$) | Partial (few $y$) | None (self-generated) | None |
| **Primary Goal** | Mapping $x \to y$ | Improving $x \to y$ with $X_u$ | Learning features $z$ | Finding patterns/clusters |
| **Data Efficiency** | Low | High | Very High | High |

---

### A2. Theoretical Foundations of Semi-Supervised Learning

#### A2.1 Key Assumptions
For unlabeled data to be useful, we must assume a relationship between the distribution of $p(x)$ and the target $p(y|x)$:
1.  **Smoothness Assumption:** If two points $x_1, x_2$ are close in a high-density region, their labels $y_1, y_2$ should be the same.
2.  **Cluster Assumption:** Points in the same cluster likely share a label. This implies the decision boundary should lie in low-density regions.
3.  **Manifold Assumption:** High-dimensional data lies on a lower-dimensional manifold. Unlabeled data helps "map" this manifold.

#### A2.2 Generative Approach — GMM with EM
We model the joint distribution $p(x, y | \theta) = p(y|\theta) p(x|y, \theta)$. For a Gaussian Mixture Model (GMM), $p(y|\theta) = w_y$ (prior) and $p(x|y, \theta) = \mathcal{N}(x; \mu_y, \Sigma_y)$.

The **log-likelihood** for the semi-supervised case is:
$$\mathcal{L}(\theta) = \sum_{i=1}^{l} \log [ p(y_i|\theta)p(x_i|y_i,\theta) ] + \sum_{i=l+1}^{l+u} \log \left( \sum_{y} p(y|\theta)p(x_i|y,\theta) \right)$$

**EM Algorithm for SSL:**
1.  **Initialize:** Train $\theta$ using only $\mathcal{D}_l$.
2.  **E-Step:** For all $x_i \in \mathcal{D}_u$, calculate the "soft label" (responsibility):
    $$\gamma_{iy} = p(y | x_i, \theta) = \frac{w_y \mathcal{N}(x_i; \mu_y, \Sigma_y)}{\sum_{j} w_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}$$
3.  **M-Step:** Update $\theta$ (means, covariances, weights) using both $D_l$ (hard labels) and $D_u$ (soft labels $\gamma_{iy}$).
4.  **Repeat** until convergence.

#### A2.3 Graph-Based Methods — Label Propagation
We represent data as a graph where nodes are samples and edges $W_{ij}$ represent similarity:
$$W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$
The algorithm "spreads" labels from $D_l$ to $D_u$ by iteratively updating the label matrix $F$:
1.  $F \leftarrow W F$ (Propagate)
2.  $F_{i} \leftarrow Y_i$ for $i \in \mathcal{D}_l$ (Clamp original labels)
This converges to a solution where labels are smooth relative to the graph structure.

#### A2.4 Self-Training and Co-Training
*   **Self-Training:** A model predicts labels for $D_u$; the most confident predictions are added to $D_l$. Risk: "Confirmation Bias" (errors are amplified).
*   **Co-Training:** Requires two "views" (independent feature sets, e.g., image and text). Model A trains on View 1 and labels $D_u$ for Model B, and vice-versa.

* **Formal pseudocode for both methods:**
Self-Training Algorithm:
─────────────────────────────────────────────
Input: Labeled set D_l, Unlabeled set D_u,
       base classifier C, threshold δ
Output: Trained classifier C*

1. Train C on D_l
2. Repeat until D_u is empty or no confident predictions:
   a. Predict labels for all x ∈ D_u using C
   b. S ← {(x, ŷ) : P(ŷ|x) ≥ δ}   # confident subset
   c. D_l ← D_l ∪ S
   d. D_u ← D_u \ S
   e. Retrain C on updated D_l
3. Return C
─────────────────────────────────────────────

Co-Training Algorithm (Blum & Mitchell, 1998):
─────────────────────────────────────────────
Assumptions:
  - x = [x^(1); x^(2)]  (two conditionally independent views)
  - Each view alone is sufficient to train a good classifier

Input: Labeled set D_l, Unlabeled pool U, learning speed k
1. L_1 = L_2 = D_l
2. Repeat until U is exhausted:
   a. Train f^(1) on L_1 using only view x^(1)
   b. Train f^(2) on L_2 using only view x^(2)
   c. Each classifier labels k most-confident samples from U
   d. f^(1)'s predictions → add to L_2
      f^(2)'s predictions → add to L_1
   e. Remove labeled samples from U
3. Return ensemble of f^(1) and f^(2)
─────────────────────────────────────────────

---

### A3. Theoretical Foundations of Self-Supervised Learning

#### A3.1 Pretext Tasks
A pretext task is a pre-designed challenge where the "ground truth" is known without human labeling:
1.  **Relative Positioning:** Predicting if patch B is to the left or right of patch A.
2.  **Rotation:** Predicting if an image was rotated $0^\circ, 90^\circ, 180^\circ,$ or $270^\circ$.
3.  **Masking:** Predicting missing pixels or words (e.g., BERT, MAE).

#### A3.2 Contrastive Learning (SimCLR)
The goal is to learn a representation space where augmented versions of the same image are close, and different images are far apart.
**InfoNCE Loss:**
$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$
*   $z_i, z_j$: Positive pair (two views of the same image).
*   $z_k$: Negative samples (other images in the batch).
*   $\tau$: Temperature hyperparameter (controls the "hardness" of negatives).

#### A3.3 Non-Contrastive Methods

While contrastive methods like SimCLR are highly effective, they rely heavily on large batches of negative samples to prevent "model collapse"—a failure mode where the encoder learns to map every input to the same constant vector, trivially achieving zero distance between positive pairs. Non-contrastive methods aim to learn useful representations using only positive pairs, employing architectural constraints rather than negative samples to avoid collapse.

#### **BYOL (Bootstrap Your Own Latent)**
BYOL (Grill et al., 2020) introduces a dual-network architecture consisting of an **Online Network** and a **Target Network**. The online network is trained to predict the target network’s representation of a different augmented view of the same image.

*   **Architecture:** The online network consists of an encoder $f_\theta$, a projector $g_\theta$, and an additional **predictor** $q_\theta$. The target network has an identical architecture (encoder $f_\xi$ and projector $g_\xi$) but lacks the predictor.
*   **Momentum Update:** The target network parameters $\xi$ are not updated via backpropagation. Instead, they are a running average of the online parameters $\theta$, controlled by a decay rate $m \in [0, 1]$ (typically $m=0.99$):
    $$\xi \leftarrow m\xi + (1-m)\theta$$
    This "slow-moving" target provides a stable representation for the online network to learn from.
*   **Loss Function:** Given two views $v$ and $v'$, the online network processes $v$ to produce a prediction $q_\theta(z_i)$, while the target network processes $v'$ to produce a target projection $z'_j$. The loss is the Mean Squared Error (MSE) between the normalized vectors:
    $$\mathcal{L}_{BYOL} = \| \bar{q}_\theta(z_i) - \text{sg}(\bar{z}'_j) \|^2$$
    where $\bar{q}$ and $\bar{z}$ denote $\ell_2$-normalized vectors and $\text{sg}(\cdot)$ is the **stop-gradient** operator.
*   **Why it Prevents Collapse:** The stop-gradient is the mathematical "anchor" of BYOL. Without it, the loss could be minimized by both networks collapsing to a constant output. By preventing gradients from flowing through the target network, the online network is forced to "bootstrap" its knowledge—it must update itself to match the target's current representation. The addition of the predictor $q_\theta$ breaks the symmetry between the two pipelines, making it impossible for a collapsed identity mapping to be an optimal solution to the prediction task.

#### **Masked Autoencoders (MAE)**
MAE (He et al., 2021) is a generative non-contrastive approach based on the Vision Transformer (ViT). It treats self-supervised learning as a reconstruction task, similar to masked language modeling in NLP (e.g., BERT).

*   **Patch Masking Procedure:** An image is divided into regular non-overlapping patches. A massive portion—typically **75%**—of these patches are randomly removed (masked). Only the remaining 25% (the visible patches) are sent to the encoder.
*   **Asymmetric Encoder-Decoder:** 
    *   The **Encoder** is a ViT that operates only on the small subset of visible patches. This makes the encoder extremely efficient.
    *   The **Decoder** is a smaller ViT that receives the full set of tokens: the encoded visible patches plus a set of learnable "mask tokens" representing the missing positions. The decoder's job is to reconstruct the original pixel values for the masked patches.
*   **Reconstruction Loss:** The model is trained using a Mean Squared Error loss calculated only on the masked pixels. Let $\mathcal{M}$ be the set of masked pixel indices, $x_p$ the original pixel value, and $\hat{x}_p$ the predicted value:
    $$\mathcal{L}_{MAE} = \frac{1}{|\mathcal{M}|} \sum_{p \in \mathcal{M}} \| x_p - \hat{x}_p \|^2$$
*   **Why High Masking Ratio?** In NLP, masking ratios are low (15%) because words are information-dense. In images, pixels are highly redundant; a pixel can be easily predicted from its immediate neighbors. By masking 75% of the image, the task becomes "hard." The model cannot rely on local interpolation; it must develop a **holistic understanding** of the objects (e.g., recognizing that if it sees a tail and a partial torso, it must reconstruct a dog). This forces the encoder to learn high-level semantic features rather than low-level texture patterns.

#### Comparison Table :

| Method | Type | Requires Labels | Key Assumption | Scalability | Typical Domain |
|---|---|---|---|---|---|
| GMM+EM | Semi | Yes (few) | Gaussian class-conditional | Low | Tabular |
| Label Propagation | Semi | Yes (few) | Graph smoothness | Medium | Small datasets |
| Self-Training | Semi | Yes (few) | Model confidence = correctness | Medium | Any |
| Co-Training | Semi | Yes (few) | View independence | Medium | Web / Multi-modal |
| SimCLR | Self | No | Augmentation invariance | High | Computer Vision |
| BYOL | Self | No | Momentum target stability | High | Computer Vision |
| MAE | Self | No | Data spatial redundancy | Very High | Vision / NLP |

---

## Part B — Skill Component

### B1. Implementation Roadmap
1.  **Split:** Partition dataset into `train_labeled`, `train_unlabeled`, and `test`.
2.  **Baseline:** Train a supervised model on `train_labeled` only.
3.  **Pretrain/SSL:** Run SSL (e.g., Label Prop) or Self-SL (e.g., SimCLR pretraining).
4.  **Downstream:** Fine-tune the model or train a linear classifier on learned features.
5.  **Evaluate:** Compare Accuracy vs. \% of labeled data used.

### B2. Code Example 1 — Label Propagation (scikit-learn)

```python
# ============================================================
# WHAT: Label Propagation for semi-supervised classification
# WHY: Demonstrates how unlabeled data improves decision boundary
# DATASET: Digits dataset (scikit-learn), 5% labeled
# EXPECTED OUTPUT: Label Propagation > Logistic Regression Accuracy
# ============================================================
import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load data
digits = datasets.load_digits()
rng = np.random.RandomState(42)
indices = np.arange(len(digits.data))
rng.shuffle(indices)

X, y = digits.data[indices], digits.target[indices]
n_labeled = 50  # Only 50 labels out of 1797

# 2. Create unlabeled set (represented by -1 in sklearn)
y_train_unlabeled = np.copy(y)
y_train_unlabeled[n_labeled:] = -1 

# 3. Supervised Baseline (on 50 samples)
clf_sup = LogisticRegression(max_iter=10000)
clf_sup.fit(X[:n_labeled], y[:n_labeled])
acc_sup = accuracy_score(y[n_labeled:], clf_sup.predict(X[n_labeled:]))

# 4. Semi-Supervised (Label Propagation)
lp_model = LabelPropagation(gamma=0.25)
lp_model.fit(X, y_train_unlabeled)
acc_ssl = accuracy_score(y[n_labeled:], lp_model.predict(X[n_labeled:]))

print(f"Supervised Accuracy (50 labels): {acc_sup:.4f}")
print(f"Semi-Supervised Accuracy (50 labels + rest unlabeled): {acc_ssl:.4f}")
```

### B3. Code Example 2 — Self-Supervised Pretraining with SimCLR (PyTorch)

```python
# ============================================================
# WHAT: Full SimCLR Pipeline (Augmentation -> Pretraining -> Linear Probe)
# WHY: Demonstrates how to learn representations without labels and evaluate them
# DATASET: CIFAR-10 (torchvision)
# EXPECTED OUTPUT: Linear probe accuracy > 50% after 5 epochs (on full dataset)
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch.nn.functional as F

# 1. SimCLR Augmentation Pipeline
class SimCLRAugmentation:
    def __init__(self, size=32):
        s = 1.0  # jitter strength
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * size) | 1),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

# 2. SimCLR Model Architecture
class SimCLRModel(nn.Module):
    def __init__(self, base_model, out_dim=128):
        super(SimCLRModel, self).__init__()
        # Use ResNet-18 backbone
        self.encoder = base_model
        # Modify first layer for 32x32 CIFAR images
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        
        dim_in = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity() # Remove classification head

        # 2-layer MLP Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

# 3. NT-Xent Loss Function
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    features = torch.cat([z_i, z_j], dim=0)
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    sim_matrix = sim_matrix / temperature
    
    # Mask self-similarity
    mask = torch.eye(2 * batch_size).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
    
    targets = torch.arange(2 * batch_size)
    targets[:batch_size] += batch_size
    targets[batch_size:] -= batch_size
    
    return F.cross_entropy(sim_matrix, targets)

# 4. Main Execution Pipeline
def run_simclr_demo():
    device = torch.device("cpu") # CPU-friendly for demo
    
    # Data Loading (Using a subset for speed in this example)
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=SimCLRAugmentation())
    # For demonstration purposes, we use a subset of 2000 images
    subset_indices = torch.arange(2000)
    train_loader = DataLoader(Subset(train_dataset, subset_indices), batch_size=64, shuffle=True)

    # Initialize Model
    resnet = models.resnet18(weights=None)
    model = SimCLRModel(resnet).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- PHASE 1: Self-Supervised Pretraining ---
    print("Starting SimCLR Pretraining (5 Epochs)...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        for (x_i, x_j), _ in train_loader:
            optimizer.zero_grad()
            _, z_i = model(x_i.to(device))
            _, z_j = model(x_j.to(device))
            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f}")

    # --- PHASE 2: Linear Probe Evaluation ---
    print("\nStarting Linear Probe Evaluation...")
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Define Linear Classifier
    classifier = nn.Linear(512, 10).to(device)
    clf_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    # Simple transform for evaluation
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    eval_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    # Train Linear Head (using representations from frozen encoder)
    model.eval()
    for epoch in range(3): # Short training for the probe
        for x, y in train_loader: # Reuse train subset for probe training
            # train_loader returns tuples of views, take the first one
            x_single = x[0].to(device)
            clf_optimizer.zero_grad()
            with torch.no_grad():
                h, _ = model(x_single)
            logits = classifier(h)
            loss = F.cross_entropy(logits, y.to(device))
            loss.backward()
            clf_optimizer.step()

    # Final Accuracy Check
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in eval_loader:
            h, _ = model(x.to(device))
            logits = classifier(h)
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y.to(device)).sum().item()

    print(f"Linear Probe Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    run_simclr_demo()

```

### B4. Common Pitfalls and Debugging Guide

| # | Pitfall | Symptom | Fix |
|---|---------|---------|-----|
| 1 | Confirmation Bias | Accuracy drops after many iterations of self-training | Use higher confidence thresholds or soft labels. |
| 2 | Model Collapse | In Self-SL, all $z$ vectors become identical; loss is low but features are useless | Use negative samples (SimCLR) or momentum encoders (BYOL). |
| 3 | Weak Augmentations | SimCLR features fail to generalize | Use aggressive color jitter and random cropping. |
| 4 | Data Leakage | Validation labels accidentally included in SSL graph | Ensure validation set is strictly held out from the graph construction. |
| 5 | Poor $\tau$ tuning | SimCLR loss doesn't converge or gradients explode | Start with $\tau=0.1$; smaller values make the task harder. |
| 6 | Out-of-dist Unlabeled Data | SSL performance is worse than supervised baseline | Filter $D_u$ to ensure it matches the distribution of $D_l$. |

### B5. Hyperparameter Tuning Guide

| Hyperparameter | Method | Recommended Range | Search Strategy | Notes |
|---|---|---|---|---|
| Temperature $\tau$ | SimCLR | 0.07–0.5 | Log-scale grid | Lower $\tau$ focuses on hard negatives. |
| Batch Size | SimCLR | 256–4096 | As large as possible | More negatives = better representations. |
| Confidence Threshold | Self-Training | 0.85–0.99 | Linear scan | High threshold prevents label noise. |
| Graph Kernel $\sigma$ | Label Prop | 0.1–10.0 | Grid search | Controls how "far" labels spread. |
| Masking Ratio | MAE | 0.5–0.8 | Fixed per domain | Vision requires higher masking (75%) than NLP. |
| Projection Head Dim | SimCLR | 128–2048 | Powers of 2 | MLP head is discarded after pretraining. |

---

## Part C — Ethical Implications & Values Component

### C1. Bias Amplification in Semi-Supervised Learning
Label propagation assumes that the graph structure correctly reflects class boundaries. If the initial labeled set $\mathcal{D}_l$ is biased (e.g., only containing samples of a specific demographic), the algorithm will propagate those labels to similar-looking unlabeled samples, effectively "colonizing" the feature space with biased labels. This creates a feedback loop where the model becomes increasingly confident in its biased predictions.

### C2. Privacy Risks
Self-supervised models are often pretrained on massive, uncurated datasets (e.g., Common Crawl). These models can "memorize" unique features of the training data. Through **Membership Inference Attacks**, an adversary can determine if a specific individual's data was used in the pretraining set, potentially exposing private information even if the model is only used for a downstream classification task.

### C3. Societal Impact
*   **Democratization:** SSL/Self-SL allows researchers in low-resource environments (e.g., developing nations, non-profits) to build powerful models without the multi-million dollar labeling budgets of Big Tech.
*   **Exclusion:** Because Self-SL relies on massive data, groups underrepresented in "standard" web data (e.g., speakers of minority languages) may find that pretrained models perform poorly for them, further widening the digital divide.

### C4. Responsible Practices Checklist
1. [ ] **Audit $D_l$:** Is the small labeled set representative of the deployment population?
2. [ ] **Distribution Check:** Does $p(x)$ for the unlabeled data match $p(x)$ for the labeled data?
3. [ ] **Augmentation Review:** Do the augmentations (e.g., color jitter) remove information critical for fairness (e.g., skin tone)?
4. [ ] **Representation Probing:** Use "linear probes" to check if the self-supervised model has learned sensitive attributes (race, gender) unintentionally.
5. [ ] **Unlabeled Data Sourcing:** Document the provenance and consent status of the unlabeled data.
6. [ ] **Baseline Comparison:** Always report the delta over a supervised baseline to ensure SSL is actually helping, not just adding complexity.

### C5. Ethical Case Study: Medical Imaging Deployment
**Scenario:** A hospital system uses a self-supervised model pretrained on 10 million public medical images to build a diagnostic classifier. It is fine-tuned on 500 labeled images from an urban hospital and deployed in rural clinics.

**Analysis:**
The primary risk is **distribution shift** combined with **representation bias**. Public datasets are often skewed toward specific demographics and imaging hardware. If the self-supervised model learns features optimized for high-end urban scanners, it may fail on lower-resolution rural equipment. 

Furthermore, if the 500 labeled images do not represent the rural population's specific health profiles (e.g., different common comorbidities), the semi-supervised mechanism (like Label Propagation) will force the rural data into "urban" categories. This could lead to a high rate of **false negatives** in rural populations, which is a critical safety failure.

**Detection & Mitigation:**
The hospital should perform **Subgroup Analysis**, evaluating performance specifically for the rural demographic before deployment. A responsible alternative is to include a small amount of labeled data from the rural site itself to "anchor" the semi-supervised learning process.

**How to Detect This Failure in Practice:**
Before deployment, the hospital team should:
1. **Performance Disaggregation:** Compute accuracy, sensitivity, and specificity separately for rural vs. urban patient records (if a small test set from rural sites is available). A >10% gap in sensitivity is a red flag.
2. **Embedding Space Analysis:** Extract representations from the self-supervised model for both urban and rural images. Use t-SNE or UMAP to visualize the embedding space. If rural images cluster in an isolated region far from labeled training data, the model has not learned to generalize to this distribution.
3. **Calibration Check:** Self-supervised models fine-tuned on small datasets can be overconfident. Use Expected Calibration Error (ECE) to check if predicted probabilities match actual accuracy. Overconfident wrong predictions in rural settings are particularly dangerous.

**Responsible Alternatives:**
1. **Domain-Adaptive SSL:** Collect even 20–30 unlabeled images from rural clinics before fine-tuning. Include these in the self-supervised pretraining step to ensure the encoder sees rural imaging characteristics.
2. **Federated Semi-Supervised Learning:** Instead of centralizing all data, train local models at each rural clinic using their own (unlabeled) data and share only model updates — protecting patient privacy while ensuring local distribution coverage.
3. **Human-in-the-Loop Deployment:** For the first 6 months of deployment, require radiologist sign-off on all model predictions at rural sites, while collecting labels for future fine-tuning.
4. **Uncertainty Quantification:** Use Monte Carlo Dropout or deep ensembles to flag predictions the model is uncertain about — these cases should automatically escalate to a human expert.
---

## Step 3 — Final Consistency Check

```
CONSISTENCY CHECK:
- Core concepts covered in Part A: SSL Assumptions, GMM-EM, Label Propagation, Pretext Tasks, Contrastive Learning (SimCLR), InfoNCE.
- Concepts testable in Module 2 (Assessment): Deriving GMM-EM updates, calculating InfoNCE loss, comparing SSL assumptions, identifying pretext tasks, analyzing co-training views, and evaluating bias amplification.
- Concepts implementable in Module 3 (Code): Label Propagation with sklearn, SimCLR loss in PyTorch, creating labeled/unlabeled splits, and linear probing.
- Ethical themes carried forward: Bias amplification in label propagation and representation bias in uncurated pretraining data.
- Notation used consistently: X_l/Y_l (labeled), X_u (unlabeled), theta (parameters), tau (temperature), sim() (similarity).
```