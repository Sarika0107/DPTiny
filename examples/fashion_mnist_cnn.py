"""Train the baseline MNIST CNN on Fashion-MNIST and evaluate it.

The model is the one from examples/mnist_cnn.py, unchanged. The images are
standardized with the training-set mean and standard deviation, the run is
seeded, and after 10 epochs the script reports the confusion matrix, the
accuracy of each class and the pair of classes confused most often.
"""

import time

import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_gpu,
    use_gpu,
)
from dptiny.backend import to_cpu, xp
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES, get_fashion_mnist
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

SEED = 42
BATCH_SIZE = 64
MAX_EPOCH = 10
LR = 0.001
NUM_CLASSES = len(CLASSES)

if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")

# Seed every random number generator DPTiny uses: weight initialization,
# dropout masks and DataLoader shuffling all draw from xp.random.
np.random.seed(SEED)
xp.random.seed(SEED)

print("Loading Fashion-MNIST dataset...")
X_train, X_test, y_train, y_test = get_fashion_mnist(flatten=False)

# Standardize with statistics of the TRAINING set only (no test leakage).
mean = float(X_train.mean(dtype=np.float64))
std = float(X_train.std(dtype=np.float64))
X_train = ((X_train - mean) / std).astype(np.float32)
X_test = ((X_test - mean) / std).astype(np.float32)
print(f"Training-set mean: {mean:.4f}, std: {std:.4f}")

if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

# Baseline model from examples/mnist_cnn.py, unchanged.
model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)
if is_gpu():
    model.to_gpu()

train_loader = DataLoader((X_train, y_train), BATCH_SIZE)
test_loader = DataLoader((X_test, y_test), BATCH_SIZE, shuffle=False)

optimizer = Adam(model, lr=LR)


def evaluate():
    """Return (accuracy, predictions) on the test set, with dropout off."""
    preds = []
    with test_mode(), no_grad():
        for x, t in test_loader:
            y = model(Variable(x))
            preds.append(to_cpu(y.data.argmax(axis=1)))
    preds = np.concatenate(preds)
    acc = float((preds == to_cpu(y_test)).mean())
    return acc, preds


start_time = time.time()
for epoch in range(MAX_EPOCH):
    model.train()
    sum_loss = 0.0
    sum_correct = 0
    train_count = 0  # separate counters, so the training loss is not
    #                  divided by the test-set size
    for x, t in train_loader:
        y = model(Variable(x))
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_correct += int((y.data.argmax(axis=1) == t).sum())
        train_count += len(t)

    train_loss = sum_loss / train_count
    train_acc = sum_correct / train_count
    test_acc, test_pred = evaluate()
    print(
        f"epoch {epoch + 1:2d}/{MAX_EPOCH} | "
        f"train loss: {train_loss:.4f} | "
        f"train acc: {train_acc:.4f} | "
        f"test acc: {test_acc:.4f} | "
        f"time: {time.time() - start_time:.1f}s"
    )

print(f"\nTraining completed in {time.time() - start_time:.1f} seconds")
print(f"Final test accuracy: {test_acc:.4f}")

# ---------------------------------------------------------------------------
# Evaluation of the final model (predictions from the last epoch's test pass)
# ---------------------------------------------------------------------------
y_true = to_cpu(y_test).astype(np.int64)
y_pred = test_pred.astype(np.int64)

# Confusion matrix: rows = true class, columns = predicted class.
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
np.add.at(cm, (y_true, y_pred), 1)

short = [name[:7] for name in CLASSES]
print("\nConfusion matrix (rows = true class, columns = predicted class):")
print(" " * 13 + " ".join(f"{s:>7}" for s in short))
for i, row in enumerate(cm):
    print(f"{CLASSES[i]:>12} " + " ".join(f"{v:7d}" for v in row))

print("\nPer-class accuracy:")
per_class = cm.diagonal() / cm.sum(axis=1)
for name, acc in zip(CLASSES, per_class):
    print(f"  {name:>12}: {acc:.4f}")

# Most confused pair, counting both directions (A->B plus B->A).
off_diag = cm.copy()
np.fill_diagonal(off_diag, 0)
both = off_diag + off_diag.T
i, j = np.unravel_index(np.argmax(np.triu(both, k=1)), both.shape)
print(
    f"\nMost confused pair: {CLASSES[i]} <-> {CLASSES[j]} "
    f"({both[i, j]} test images: {off_diag[i, j]} {CLASSES[i]} predicted as "
    f"{CLASSES[j]}, {off_diag[j, i]} {CLASSES[j]} predicted as {CLASSES[i]})"
)

# Largest single direction, for completeness.
a, b = np.unravel_index(np.argmax(off_diag), off_diag.shape)
print(
    f"Most frequent single mistake: {CLASSES[a]} predicted as {CLASSES[b]} "
    f"({off_diag[a, b]} test images)"
)
