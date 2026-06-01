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
         CIFAR-10 (downloaded automatically via torchvision, ~170 MB)

Run: python coding_boilerplate.py

Note: All TODOs raise NotImplementedError. Implement each TODO in order,
then call validate_student_work() to check your implementations.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tv_datasets, transforms, models

# Version checks
import sklearn
assert sklearn.__version__.startswith("1."), \
    f"Expected scikit-learn 1.x, got {sklearn.__version__}"
assert int(torch.__version__.split(".")[0]) >= 2, \
    f"Expected PyTorch 2.x, got {torch.__version__}"

# =============================================================================
# CONFIGURATION — All hyperparameters live here. Never use magic numbers.
# =============================================================================
class CFG:
    # Data
    labeled_fraction: float = 0.10   # fraction of training data that is labeled
    test_size: float = 0.20
    random_seed: int = 42

    # Label Propagation
    lp_kernel: str = "knn"
    lp_n_neighbors: int = 7
    lp_max_iter: int = 1000

    # Logistic Regression Baseline
    lr_max_iter: int = 10000
    lr_solver: str = "lbfgs"
    lr_multi_class: str = "multinomial"

    # Contrastive Learning (SimCLR)
    batch_size: int = 128
    temperature: float = 0.07
    embedding_dim: int = 128
    contrastive_epochs: int = 5        # kept low for assignment runtime
    contrastive_lr: float = 1e-3

    # Paths
    output_dir: str = "./outputs"
    data_dir: str = "./data"


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_digits_data():
    """
    Load the scikit-learn Digits dataset and split into train/test.

    Returns:
        X_train, X_test: numpy arrays of shape (n_train, 64) and (n_test, 64)
        y_train, y_test: numpy arrays of integer labels (0–9)
    """
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG.test_size,
        random_state=CFG.random_seed,
        stratify=y
    )
    print(f"[Data] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[Data] Classes: {np.unique(y_train)}")
    return X_train, X_test, y_train, y_test


def log_results(results: dict) -> None:
    """
    Print a formatted table of method -> accuracy results.

    Args:
        results: dict mapping method name (str) to accuracy (float)
    """
    print("\n" + "=" * 50)
    print(f"{'Method':<35} {'Accuracy':>10}")
    print("=" * 50)
    for method, acc in results.items():
        print(f"{method:<35} {acc:>10.4f}")
    print("=" * 50 + "\n")


def plot_comparison(results: dict, save_path: str = None) -> None:
    """
    Plot a bar chart comparing method accuracies.

    Args:
        results: dict mapping method name (str) to accuracy (float)
        save_path: if provided, save figure to this path instead of displaying
    """
    os.makedirs(CFG.output_dir, exist_ok=True)
    methods = list(results.keys())
    accs = list(results.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, accs, color=["steelblue", "tomato", "seagreen"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Method Comparison: Supervised vs Semi-Supervised")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center", va="bottom", fontsize=11)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(CFG.output_dir, "method_comparison.png")
    plt.savefig(save_path, dpi=100)
    print(f"[Plot] Saved to {save_path}")
    plt.close()


# =============================================================================
# SECTION 1: Label Mask Construction
# =============================================================================

def create_label_mask(n_train: int) -> np.ndarray:
    """
    Create a boolean mask indicating which training samples are labeled.

    Args:
        n_train: total number of training samples

    Returns:
        label_mask: boolean numpy array of shape (n_train,)
    """
    # ─────────────────────────────────────────────────────────────────────────
    # TODO [T1]: Create the label mask
    # Bloom's Level: Apply
    # Difficulty: Easy | Expected lines: ~3
    # Description:
    #   Create a boolean array `label_mask` of shape (n_train,) where True
    #   indicates a labeled sample. Use CFG.labeled_fraction to determine how
    #   many samples are labeled. Labeled samples should be the FIRST
    #   int(n_train * CFG.labeled_fraction) entries (index 0 to l-1).
    #   The remaining entries should be False (treated as unlabeled).
    # Hints:
    #   1. (Conceptual) In SSL, we simulate limited labels by treating most of
    #      the training set as unlabeled. The mask encodes which indices we
    #      "reveal" to the algorithm.
    #   2. (Implementation) np.zeros(n, dtype=bool) creates an all-False boolean
    #      array. Set the first int(n_train * CFG.labeled_fraction) entries to True.
    # Expected behavior: label_mask.sum() == int(n_train * CFG.labeled_fraction)
    # ─────────────────────────────────────────────────────────────────────────
    # >>> YOUR CODE HERE <<<

    raise NotImplementedError("TODO T1 not yet implemented")

    # >>> END YOUR CODE <<<


# =============================================================================
# SECTION 2: Supervised Baseline
# =============================================================================

def train_supervised_baseline(X_train: np.ndarray,
                               y_train: np.ndarray,
                               label_mask: np.ndarray,
                               X_test: np.ndarray,
                               y_test: np.ndarray) -> float:
    """
    Train a logistic regression classifier on the LABELED portion only.

    Args:
        X_train: full training features, shape (n_train, n_features)
        y_train: full training labels, shape (n_train,)
        label_mask: boolean mask identifying labeled samples
        X_test: test features
        y_test: test labels

    Returns:
        accuracy: float accuracy on the test set
    """
    # ─────────────────────────────────────────────────────────────────────────
    # TODO [T2]: Train the supervised baseline
    # Bloom's Level: Apply
    # Difficulty: Easy | Expected lines: ~5
    # Description:
    #   Use label_mask to extract only the labeled training samples. Train a
    #   LogisticRegression model using CFG.lr_solver, CFG.lr_max_iter, and
    #   CFG.lr_multi_class. Evaluate on X_test/y_test and return accuracy.
    #   This baseline only sees CFG.labeled_fraction of the training data.
    # Hints:
    #   1. (Conceptual) This baseline establishes a lower bound. Any SSL method
    #      that doesn't outperform this is not leveraging unlabeled data.
    #   2. (Implementation) Use X_train[label_mask] and y_train[label_mask]
    #      to extract labeled samples. Use accuracy_score() for evaluation.
    # Expected behavior: returns a float in [0, 1]; typically 0.70–0.85 for
    #   10% labels on the Digits dataset.
    # ─────────────────────────────────────────────────────────────────────────
    # >>> YOUR CODE HERE <<<

    raise NotImplementedError("TODO T2 not yet implemented")

    # >>> END YOUR CODE <<<


# =============================================================================
# SECTION 3: Semi-Supervised Learning — Label Propagation
# =============================================================================

def run_label_propagation(X_train: np.ndarray,
                           y_train: np.ndarray,
                           label_mask: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray) -> float:
    """
    Run Label Propagation semi-supervised classification.

    Args:
        X_train: full training features (labeled + unlabeled)
        y_train: full training labels (unlabeled entries will be masked to -1)
        label_mask: boolean mask — True = labeled, False = unlabeled
        X_test: test features
        y_test: test labels

    Returns:
        accuracy: float accuracy on the test set
    """
    # ─────────────────────────────────────────────────────────────────────────
    # TODO [T3]: Implement Label Propagation
    # Bloom's Level: Apply
    # Difficulty: Medium | Expected lines: ~8
    # Description:
    #   (Step 1) Create y_partial: a copy of y_train where entries where
    #     label_mask is False are replaced with -1. sklearn's LabelPropagation
    #     uses -1 as the sentinel for unlabeled samples.
    #   (Step 2) Instantiate a LabelPropagation model using CFG.lp_kernel,
    #     CFG.lp_n_neighbors, and CFG.lp_max_iter.
    #   (Step 3) Fit on ALL of X_train (labeled + unlabeled) with y_partial.
    #   (Step 4) Predict on X_test and compute accuracy.
    # Hints:
    #   1. (Conceptual) LabelPropagation "spreads" labels through the kNN graph.
    #      Unlabeled nodes receive labels by proximity to labeled neighbors.
    #      Using ALL training data (including unlabeled) is the key difference
    #      from supervised learning.
    #   2. (Implementation) y_partial = y_train.copy(); y_partial[~label_mask] = -1
    #      The ~ operator inverts a boolean mask.
    # Expected behavior: returns float typically in range 0.88–0.97 for knn, k=7,
    #   10% labels on Digits dataset.
    # ─────────────────────────────────────────────────────────────────────────
    # >>> YOUR CODE HERE <<<

    raise NotImplementedError("TODO T3 not yet implemented")

    # >>> END YOUR CODE <<<


# =============================================================================
# SECTION 4: Evaluation and Comparison
# =============================================================================

def evaluate_and_compare(baseline_acc: float,
                          lp_acc: float) -> dict:
    """
    Compare supervised baseline and Label Propagation, print and plot results.

    Args:
        baseline_acc: accuracy of supervised baseline
        lp_acc: accuracy of label propagation

    Returns:
        results: dict mapping method name to accuracy
    """
    # ─────────────────────────────────────────────────────────────────────────
    # TODO [T4]: Evaluate and compare methods
    # Bloom's Level: Analyze
    # Difficulty: Medium | Expected lines: ~10
    # Description:
    #   (Step 1) Build a results dict: {"Supervised Baseline (10% labels)": baseline_acc,
    #     "Label Propagation (10% labels)": lp_acc}
    #   (Step 2) Call log_results(results) to print the table.
    #   (Step 3) Call plot_comparison(results) to generate the bar chart.
    #   (Step 4) Compute and print the absolute improvement:
    #     improvement = lp_acc - baseline_acc
    #     Print whether SSL helped or hurt, and by how much.
    #   (Step 5) Return the results dict.
    # Hints:
    #   1. (Conceptual) If Label Propagation underperforms the baseline,
    #      the manifold/cluster assumption may not hold for this data.
    #      Think about why the Digits dataset is (or isn't) a good fit.
    #   2. (Implementation) f-strings are your friend for formatted output.
    #      Use f"{improvement:+.4f}" to show sign explicitly.
    # Expected behavior: prints a table and saves a bar chart PNG.
    # ─────────────────────────────────────────────────────────────────────────
    # >>> YOUR CODE HERE <<<

    raise NotImplementedError("TODO T4 not yet implemented")

    # >>> END YOUR CODE <<<


# =============================================================================
# SECTION 5: Self-Supervised Learning — Contrastive Augmentation
# =============================================================================

class SimCLRAugmentation:
    """
    Applies two independent random augmentations to the same image,
    producing a positive pair (x_i, x_j) for contrastive learning.
    """

    def __init__(self, image_size: int = 32):
        """
        Pre-completed: augmentation pipeline is defined here.
        Students should read and understand this pipeline — it will be
        referenced in TODO T6 and in Module 2 assessment questions.
        """
        s = 1.0  # color jitter strength
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)],
                p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])

    def __call__(self, x):
        """Return two independently augmented views of the same image."""
        return self.transform(x), self.transform(x)


def load_cifar10_for_contrastive():
    """
    Load CIFAR-10 with SimCLR augmentation for contrastive pretraining.
    Uses a subset of 2000 images to keep runtime acceptable on CPU.

    Returns:
        train_loader: DataLoader yielding ((x_i, x_j), label) tuples
    """
    os.makedirs(CFG.data_dir, exist_ok=True)
    dataset = tv_datasets.CIFAR10(
        root=CFG.data_dir,
        train=True,
        download=True,
        transform=SimCLRAugmentation(image_size=32)
    )
    # Subset to 2000 samples for CPU-friendly runtime
    subset = Subset(dataset, indices=list(range(2000)))
    loader = DataLoader(
        subset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    print(f"[Data] CIFAR-10 contrastive loader: {len(subset)} samples, "
          f"batch_size={CFG.batch_size}")
    return loader


# =============================================================================
# SECTION 6: NT-Xent (InfoNCE) Loss
# =============================================================================

def nt_xent_loss(z_i: torch.Tensor,
                 z_j: torch.Tensor,
                 temperature: float = None) -> torch.Tensor:
    """
    Compute the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
    for a batch of contrastive pairs.

    Args:
        z_i: embeddings of view 1, shape (N, D), L2-normalized
        z_j: embeddings of view 2, shape (N, D), L2-normalized
        temperature: scaling factor tau; defaults to CFG.temperature

    Returns:
        loss: scalar tensor
    """
    if temperature is None:
        temperature = CFG.temperature

    # ─────────────────────────────────────────────────────────────────────────
    # TODO [T6]: Implement the NT-Xent loss
    # Bloom's Level: Analyze
    # Difficulty: Hard | Expected lines: ~15
    # Description:
    #   (Step 1) Concatenate z_i and z_j along dimension 0:
    #     features = torch.cat([z_i, z_j], dim=0)  → shape (2N, D)
    #   (Step 2) Compute the (2N x 2N) cosine similarity matrix.
    #     Use F.cosine_similarity with broadcasting:
    #     sim = F.cosine_similarity(features.unsqueeze(1),
    #                               features.unsqueeze(0), dim=2)
    #   (Step 3) Scale by temperature: sim = sim / temperature
    #   (Step 4) Mask out self-similarities (diagonal) by setting them to a
    #     large negative value (-9e15) so they are excluded from softmax.
    #     mask = torch.eye(2 * N, dtype=torch.bool, device=z_i.device)
    #     sim = sim.masked_fill(mask, -9e15)
    #   (Step 5) Construct labels. The positive pair for index i (0 to N-1)
    #     is index i+N, and vice versa.
    #     labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)])
    #   (Step 6) Compute cross-entropy loss: F.cross_entropy(sim, labels)
    # Hints:
    #   1. (Conceptual) The loss treats the 2N embeddings as a classification
    #      problem: given embedding z_i, identify its positive pair z_j among
    #      all 2N-1 other embeddings. Temperature tau controls how peaked the
    #      softmax is — low tau forces the model to be very discriminative.
    #   2. (Implementation) Make sure labels and sim are on the same device.
    #      Use .to(z_i.device) for any newly created tensors.
    # Expected behavior: returns a scalar tensor. Typical range: 4.0–6.5 at
    #   initialization for batch_size=128 on CIFAR-10.
    # ─────────────────────────────────────────────────────────────────────────
    # >>> YOUR CODE HERE <<<

    raise NotImplementedError("TODO T6 not yet implemented")

    # >>> END YOUR CODE <<<


# =============================================================================
# SECTION 7: SimCLR Model
# =============================================================================

class SimCLRModel(nn.Module):
    """ResNet-18 backbone with a 2-layer MLP projection head for SimCLR."""

    def __init__(self, embedding_dim: int = None):
        """Pre-completed: backbone and projection head are defined here."""
        super().__init__()
        if embedding_dim is None:
            embedding_dim = CFG.embedding_dim

        # Backbone: ResNet-18 without pretrained weights
        backbone = models.resnet18(weights=None)
        # Adapt for 32x32 CIFAR images (original is designed for 224x224)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        dim_in = backbone.fc.in_features  # 512 for ResNet-18
        backbone.fc = nn.Identity()  # Remove classification head
        self.encoder = backbone

        # 2-layer MLP projection head (discarded after pretraining)
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, embedding_dim)
        )
        self.dim_in = dim_in

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input images, shape (N, 3, H, W)
        Returns:
            h: encoder representations, shape (N, 512) — used for linear probe
            z: projected embeddings, shape (N, embedding_dim) — used for loss
        """
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=-1)  # L2-normalize for cosine similarity
        return h, z


# =============================================================================
# SECTION 7 (continued): Embedding Validation
# =============================================================================

def validate_embeddings(model: SimCLRModel,
                         loader: DataLoader,
                         device: torch.device) -> float:
    """
    Validate self-supervised embeddings using nearest-neighbor accuracy.
    A useful sanity check: if the representations are meaningful, embeddings
    from the same class should be closer than embeddings from different classes.

    Args:
        model: trained SimCLRModel
        loader: DataLoader yielding ((x_i, x_j), label) tuples
        device: compute device

    Returns:
        nn_accuracy: float, nearest-neighbor classification accuracy
    """
    # ─────────────────────────────────────────────────────────────────────────
    # TODO [T7]: Design a validation function for self-supervised embeddings
    # Bloom's Level: Create
    # Difficulty: Hard | Expected lines: ~25
    # Description:
    #   This task asks you to DESIGN a validation metric from scratch.
    #   Nearest-Neighbor (NN) accuracy is a standard self-supervised evaluation:
    #   for each embedding, find its nearest neighbor in the embedding space
    #   (excluding itself), and check if they share the same class label.
    #
    #   (Step 1) Set model to eval mode. Extract all embeddings and labels:
    #     - For each batch (x_pair, label) in loader, take x_pair[0] (first view)
    #     - Pass through model.encoder (not the projector) to get h
    #     - Collect all h into a list, convert to tensor after the loop
    #   (Step 2) L2-normalize all embeddings.
    #   (Step 3) Compute the full pairwise cosine similarity matrix
    #     (shape: n_samples x n_samples).
    #   (Step 4) Mask the diagonal (self-similarity) to -inf so each sample's
    #     nearest neighbor is not itself.
    #   (Step 5) For each sample, find argmax of its row (nearest neighbor index).
    #     Check if labels[i] == labels[nearest_neighbor[i]].
    #   (Step 6) Return the fraction of correct NN matches.
    # Hints:
    #   1. (Conceptual) NN accuracy measures whether the encoder has learned a
    #      meaningful metric space without any classifier training. A random
    #      encoder gives ~10% NN accuracy on CIFAR-10 (10 classes). After
    #      5 epochs of SimCLR pretraining, expect 20–40% on the 2000-sample subset.
    #   2. (Implementation) Keep all tensors on CPU for this function to avoid
    #      memory issues with the full similarity matrix. Use:
    #      all_h = torch.cat(h_list, dim=0)  after collecting embeddings.
    # Expected behavior: returns a float in [0, 1]. Print it for the student.
    # ─────────────────────────────────────────────────────────────────────────
    # >>> YOUR CODE HERE <<<

    raise NotImplementedError("TODO T7 not yet implemented")

    # >>> END YOUR CODE <<<


# =============================================================================
# VALIDATION HARNESS — Call this to check your work
# =============================================================================

def validate_student_work(label_mask: np.ndarray,
                           lp_accuracy: float,
                           baseline_accuracy: float,
                           loss_values: list) -> None:
    """
    Call this function to verify your implementations before submitting.
    All assertions should pass if your implementations are correct.

    Args:
        label_mask: output of create_label_mask()
        lp_accuracy: output of run_label_propagation()
        baseline_accuracy: output of train_supervised_baseline()
        loss_values: list of per-epoch NT-Xent loss values from pretraining
    """
    print("\n" + "=" * 50)
    print("VALIDATION HARNESS")
    print("=" * 50)

    # Check T1
    assert label_mask.dtype == bool, \
        f"label_mask must be boolean dtype, got {label_mask.dtype}"
    n_train_expected = label_mask.shape[0]
    expected_labeled = int(n_train_expected * CFG.labeled_fraction)
    assert label_mask.sum() == expected_labeled, \
        (f"Expected {expected_labeled} labeled samples, "
         f"got {label_mask.sum()}")
    print(f"✓ T1: label_mask correct — {label_mask.sum()} labeled samples "
          f"({CFG.labeled_fraction * 100:.0f}% of {n_train_expected})")

    # Check T3
    assert lp_accuracy > 0.5, \
        (f"Label Propagation accuracy {lp_accuracy:.3f} is below random "
         f"chance (>0.5 expected for 10-class Digits)")
    print(f"✓ T3: Label Propagation accuracy = {lp_accuracy:.4f}")

    # Check T4
    assert lp_accuracy > baseline_accuracy - 0.05, \
        ("SSL should not perform drastically worse (>5%) than supervised "
         f"baseline. Got LP={lp_accuracy:.3f}, baseline={baseline_accuracy:.3f}")
    print(f"✓ T4: Comparative evaluation passed — "
          f"LP={lp_accuracy:.4f}, Baseline={baseline_accuracy:.4f}, "
          f"Δ={lp_accuracy - baseline_accuracy:+.4f}")

    # Check T6
    assert len(loss_values) > 0, \
        "loss_values is empty — T6 pretraining loop did not record losses"
    assert loss_values[-1] < loss_values[0] * 1.5, \
        (f"NT-Xent loss is not converging: "
         f"initial={loss_values[0]:.4f}, final={loss_values[-1]:.4f}")
    print(f"✓ T6: NT-Xent loss — "
          f"initial={loss_values[0]:.4f}, final={loss_values[-1]:.4f}")

    print("\n✅ All validation checks passed!")
    print("=" * 50 + "\n")


# =============================================================================
# PRETRAINING LOOP (pre-completed skeleton — uses your T6 nt_xent_loss)
# =============================================================================

def pretrain_simclr(model: SimCLRModel,
                    loader: DataLoader,
                    device: torch.device) -> list:
    """
    Run the SimCLR self-supervised pretraining loop.
    This function is pre-completed. It calls your nt_xent_loss (T6).

    Args:
        model: SimCLRModel instance
        loader: DataLoader yielding ((x_i, x_j), label) tuples
        device: compute device

    Returns:
        loss_history: list of average loss per epoch
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=CFG.contrastive_lr)
    loss_history = []

    print(f"\n[SimCLR] Starting pretraining for {CFG.contrastive_epochs} epochs...")
    for epoch in range(CFG.contrastive_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (x_i, x_j), _ in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        print(f"  Epoch [{epoch + 1}/{CFG.contrastive_epochs}] "
              f"Avg Loss: {avg_loss:.4f}")

    return loss_history


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Orchestrates the full assessment pipeline.
    Students implement the TODO functions; this function calls them in order.
    """
    print("=" * 60)
    print("CSE 329 | Semi-Supervised & Self-Supervised Learning")
    print("=" * 60)
    set_seed(CFG.random_seed)
    os.makedirs(CFG.output_dir, exist_ok=True)

    # ── Part 1: Semi-Supervised Pipeline ──────────────────────────────────
    print("\n[PART 1] Semi-Supervised Learning with Label Propagation")
    print("-" * 60)
    X_train, X_test, y_train, y_test = load_digits_data()
    n_train = len(X_train)

    # T1: Create label mask
    label_mask = create_label_mask(n_train)

    # T2: Supervised baseline
    print(f"\n[Baseline] Training on {label_mask.sum()} labeled samples...")
    baseline_acc = train_supervised_baseline(
        X_train, y_train, label_mask, X_test, y_test
    )
    print(f"[Baseline] Accuracy: {baseline_acc:.4f}")

    # T3: Label Propagation
    print(f"\n[LP] Running Label Propagation on all {n_train} samples...")
    lp_acc = run_label_propagation(
        X_train, y_train, label_mask, X_test, y_test
    )
    print(f"[LP] Accuracy: {lp_acc:.4f}")

    # T4: Evaluate and compare
    results = evaluate_and_compare(baseline_acc, lp_acc)

    # ── Part 2: Self-Supervised Pipeline ──────────────────────────────────
    print("\n[PART 2] Self-Supervised Learning with SimCLR")
    print("-" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # T5: Load CIFAR-10 with contrastive augmentation (pre-completed)
    contrastive_loader = load_cifar10_for_contrastive()

    # T6: NT-Xent loss is called inside pretrain_simclr
    model = SimCLRModel(embedding_dim=CFG.embedding_dim).to(device)
    loss_history = pretrain_simclr(model, contrastive_loader, device)

    # T7: Validate embeddings
    print("\n[T7] Running embedding validation (NN accuracy)...")
    nn_acc = validate_embeddings(model, contrastive_loader, device)
    print(f"[T7] Nearest-Neighbor Accuracy after pretraining: {nn_acc:.4f}")

    # ── Validation Harness ────────────────────────────────────────────────
    validate_student_work(label_mask, lp_acc, baseline_acc, loss_history)


if __name__ == "__main__":
    main()
