# 🧠 Self-Pruning Neural Network (CIFAR-10)

## 📌 Overview

This project implements a **self-pruning neural network** that learns to remove unimportant weights *during training* using a learnable gating mechanism.

Unlike traditional pruning (post-training), this model dynamically adapts its architecture, producing a **sparse and efficient network**.

---

## 🚀 Key Highlights

* Custom `PrunableLinear` layer (from scratch)
* Learnable gating mechanism using sigmoid
* L1 sparsity regularization
* Dynamic pruning during training
* Accuracy vs Sparsity trade-off analysis
* CIFAR-10 image classification

---

## 🧠 Core Idea

Each weight is controlled by a learnable gate:

```python
pruned_weight = weight * sigmoid(gate_score)
```

* Gate ≈ 0 → weight is pruned
* Gate ≈ 1 → weight is retained

### Loss Function

```python
Total Loss = CrossEntropyLoss + λ * L1(gates)
```

* L1 regularization forces many gates → 0
* λ controls pruning strength

---

## 🏗️ Architecture

A feedforward neural network built using multiple custom `PrunableLinear` layers.

Each layer:

* Learns weights and gate scores
* Applies sigmoid to generate gates
* Uses pruned weights during forward pass

---

## 📂 Project Structure

```
├── self_pruning_model.ipynb
├── README.md
├── REPORT.md
├── requirements.txt
├── results/
│   └── lambda_comparison.png
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Manasv-i/self-pruning-neural-network
cd self-pruning-network
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
jupyter notebook self_pruning_model.ipynb
```

---

## 📊 Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
| ---------- | ----------------- | ------------ |
| **1e-05**  | 49.27             | 38.33        |
| **0.0001** | 46.59             | 46.55        |
| **0.001**  | 42.56             | 57.12        |

---

## 📈 Key Insights

* **Low λ (1e-05)**
  → Highest accuracy (49.27%)
  → Least pruning (38.33%)

* **Medium λ (0.0001)**
  → Balanced trade-off
  → Good sparsity (46.55%) with moderate accuracy

* **High λ (0.001)**
  → Maximum pruning (57.12%)
  → Accuracy drops (42.56%)

👉 This clearly demonstrates the **sparsity vs performance trade-off**

---

## 📉 Visualization

The model generates:

* Training loss curves
* Accuracy vs sparsity comparison
* Gate value distribution

Saved as:

```
results/lambda_comparison.png
```

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

---

## 🎯 Case Study Requirements Covered

✔ Custom PrunableLinear implementation
✔ Gate-based pruning mechanism
✔ L1 sparsity loss
✔ Training with multiple λ values
✔ Accuracy + sparsity evaluation
✔ Visualization of gate distribution

---

## 💡 Future Improvements

* Structured pruning (neurons/channels)
* Hard-concrete gates for sharper sparsity
* FastAPI deployment (API inference)
* Model compression benchmarking

---

## 👩‍💻 Author

**Manasvi Agarwal**
102303680
B.E. Computer Engineering (2027)

---

## ⭐ Conclusion

The model successfully learns to **self-prune during training**, achieving up to **57% sparsity** while maintaining reasonable accuracy, demonstrating an effective approach to **dynamic model compression**.
