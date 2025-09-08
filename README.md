# Not‑Scikit‑Learn
**From‑scratch ML + visual SVMs, verified against scikit‑learn/PyTorch.**  

---

## 30‑second overview
This repo re‑implements several classic ML models **from first principles** and backs them up with **pytest parity checks** against reference libraries. A small SVM visualizer turns the kernel trick into pictures you can discuss in an interview.

**Review in 2–3 minutes**
1) Skim `models/` for concise, framework‑free implementations.  
2) Open `test_*.py` to see parity/sanity checks.  
3) Run the SVM demo (`python SVM_Visualizer/main.py`) and glance at the PNGs in `SVM_Visualizer/results/`.

---

## Quickstart

```bash
# Python 3.11+
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Tests (skip NN parity tests if PyTorch is troublesome)
pytest -q
pytest -q -k "not neural_networks"

# Visual demos (figures saved to SVM_Visualizer/results/)
python SVM_Visualizer/main.py
```
**Data** lives in `data/` (e.g., `wine.txt`, `digits.csv`, `german_numerical-binsensitive.csv`, plus small 2‑D toy CSVs).

---

## What each model demonstrates

### Linear Regression — `models/LinearRegression.py`
- **Shows:** how the closed‑form **normal equation** solves regression .
- **How:** pseudo‑inverse with explicit bias handling (append a column of 1s). Loss is average MSE.

### Multiclass Logistic Regression — `models/LogisticRegression.py`
- **Shows:** softmax + cross‑entropy with **mini‑batch SGD** and simple convergence logic.
- **How:** explicit softmax and gradient updates; accuracy helper to track progress.

### Bernoulli Naive Bayes — `models/NaiveBayes.py`
- **Shows:** independence assumption in action on **0/1 features**, with **Laplace smoothing** on priors and conditionals.
- **How:** multiplies class prior by feature likelihoods (clear math, easy to audit).  

### K‑Means (with a simple classifier wrapper) — `models/KMeans.py`
- **Shows:** the classic **assign → update** loop, tolerance‑based convergence, and practical edge cases.
- **How:** random centroid seeding; empty clusters are re‑seeded; majority‑label mapping turns clusters into a classifier for evaluation.

### Tiny Neural Networks — `models/NeuralNetwork.py`
- **Shows:** forward pass, derivatives, and **manual backprop** without a framework.
- **How:** `OneLayerNN` (linear); `TwoLayerNN` (1 hidden layer with ReLU/Sigmoid) trained with SGD; parity check vs. a matching PyTorch MLP.

---

## Visual SVMs — why the kernel trick “clicks” here
Run `python SVM_Visualizer/main.py`; PNGs are written to `SVM_Visualizer/results/`.

1) **Feature‑space embedding** — map \(x_1,x_2\) → \(x_1,x_2,x_1^2+x_2^2\). A **linear** SVM there becomes a **circular** boundary back in 2‑D.  
2) **Kernel vs. explicit features** — a kernel SVM with \(K(x,x') = x\cdot x' + \|x\|^2\,\|x'\|^2\) replicates the linear SVM trained on the embedded data.  
3) **Support‑vector evolution** — retrain on only support vectors to reveal which points truly define the margin.

---

## Tests & what “correct” means here
- **Linear Regression**: predictions align with scikit‑learn’s `LinearRegression` (consistent intercept handling).  
- **Logistic Regression**: multiclass accuracy on the same split matches scikit‑learn’s `LogisticRegression` within tolerance (softmax + SGD; comparable setup)
- **Naive Bayes**: accuracy on the preprocessed German Credit set matches `BernoulliNB` within a tight tolerance.  
- **K‑Means**: reliably outperforms a naive baseline and lands near scikit‑learn’s `KMeans` (given random init).  
- **Neural Nets**: the 2‑layer model’s error is comparable to a same‑shape PyTorch MLP trained with a matching loop.

---

## Repository map (for a quick skim)

```
data/                # wine.txt, digits.csv, german credit (binary), toy 2‑D CSVs
models/              # core implementations (small, documented)
SVM_Visualizer/      # decision boundaries, kernels, support vectors → saves PNGs to results/
test_*.py            # parity + sanity tests
requirements.txt
```

---
