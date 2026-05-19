| Day    | Main Goal                                       |
| ------ | ----------------------------------------------- |
| Day 1  | Dataset preprocessing and session construction  |
| Day 2  | Build directed session graphs                   |
| Day 3  | Implement GGNN from scratch                     |
| Day 4  | Implement multi-head self-attention             |
| Day 5  | Complete temporal feature evolution module      |
| Day 6  | Build user-app bipartite graph                  |
| Day 7  | Implement GCN from scratch                      |
| Day 8  | Add self-attention to personalized module       |
| Day 9  | Complete personalized feature extraction module |
| Day 10 | Implement feature fusion module                 |
| Day 11 | Implement prediction layer                      |
| Day 12 | Implement loss function and optimizer           |
| Day 13 | Build complete training pipeline                |
| Day 14 | Implement evaluation metrics                    |
| Day 15 | Train and validate full model                   |
| Day 16 | Implement baseline models                       |
| Day 17 | Perform debugging and tensor validation         |
| Day 18 | Conduct ablation studies                        |
| Day 19 | Hyperparameter tuning                           |
| Day 20 | Reproduce paper results                         |
| Day 21 | Refactor and finalize codebase                  |



This paper is actually a **full research pipeline**, not just “implement one model.”
You need to implement:

* Dataset pipeline
* Session construction
* Two graph systems
* GGNN
* Self-attention
* GCN
* Feature fusion
* Training pipeline
* Evaluation pipeline
* Ablation system

The paper structure itself already gives you the implementation order. 

---

# First: Understand the REAL Architecture

Before coding, mentally split the paper into 7 engineering blocks.

```text
RAW APP LOGS
    ↓
Session Builder
    ↓
Graph Builder
    ↓
Temporal Module
    ├── GGNN
    └── Self Attention
    ↓
Personalized Module
    ├── GCN
    └── Self Attention
    ↓
Fusion Module
    ↓
Prediction Layer
    ↓
Loss + Training
```

---

# Your Goal

You should NOT implement the whole paper at once.

You should implement:

1. Data pipeline first
2. One graph first
3. One neural module first
4. Then combine slowly

Otherwise debugging becomes impossible.

---

# Recommended Tech Stack

Use:

```text
Python
PyTorch
PyTorch Geometric (optional but recommended)
NumPy
Pandas
NetworkX (optional for debugging)
```

---

# FULL ROADMAP (Day-by-Day)

---

# WEEK 1 — Foundation + Data Pipeline

---

# Day 1 — Reproduce the Dataset Pipeline

### One-line goal

Understand and implement preprocessing exactly like the paper.

---

# What you should do

Read these sections carefully:

* Problem definition
* Dataset preprocessing
* Session construction

Paper sections:  and 

---

# Your coding target

Implement:

```python
load_data()
clean_data()
build_sessions()
train_val_test_split()
```

---

# Important concepts

The paper converts raw logs into:

```text
(user, app, timestamp)
```

Then groups them into sessions.

A session:

```text
[YouTube, Chrome, WhatsApp, Spotify]
```

---

# Critical preprocessing

Paper says:

* Remove users with <20 interactions
* Remove apps with <5 usages
* Session window = 300 seconds

This is VERY important. 

---

# Output of Day 1

You should have:

```python
sessions = [
    [4, 10, 2, 7],
    [5, 1, 3],
    ...
]
```

AND:

```python
train_sessions
val_sessions
test_sessions
```

---

# Day 2 — Build Directed Session Graphs

### One-line goal

Convert sessions into graphs.

---

# Read

Graph construction module. 

---

# Understand deeply

Session:

```text
A → B → C → D
```

Becomes:

```text
Nodes = apps
Edges = transitions
```

---

# Build

For each session:

```python
edge_index
adjacency_matrix
```

Example:

```python
session = [1,2,3,2]

edges:
1→2
2→3
3→2
```

---

# Important

Paper uses:

```text
A_out
A_in
```

Then concatenates them. 

So implement:

```python
A = [A_out | A_in]
```

---

# Output of Day 2

You should be able to visualize a session graph.

---

# Day 3 — Implement GGNN From Scratch

### One-line goal

Implement temporal propagation.

---

# Read

GGNN equations carefully. 

---

# What GGNN does intuitively

Normal GNN:

```text
neighbor aggregation only
```

GGNN:

```text
neighbor aggregation
+
memory/update gates
```

So it behaves like:

```text
GNN + GRU
```

---

# You must implement

These equations exactly:

z_t = \sigma(W_z c_t + P_z s_{t-1})

r_t = \sigma(W_r c_t + P_r s_{t-1})

\hat{h}*t = \tanh(W_h c_t + P_h(r_t \odot s*{t-1}))

h_t = (1-z_t) \odot s_{t-1} + z_t \odot \hat{h}_t

---

# Your implementation target

Create:

```python
class GGNN(nn.Module)
```

---

# DO NOT use PyTorch Geometric first

Implement manually first.

Why?

Because otherwise you won’t understand:

* message passing
* propagation
* gated updates

---

# Output of Day 3

Feed one graph → get updated node embeddings.

---

# Day 4 — Implement Self Attention

### One-line goal

Add long-term dependency modeling.

---

# Read

Attention equations. 

---

# Important intuition

GGNN captures:

```text
local transitions
```

Self-attention captures:

```text
global relationships
```

Example:

```text
YouTube used earlier
→ later Spotify usage
```

Attention connects them directly.

---

# Implement

```python
class MultiHeadSelfAttention(nn.Module)
```

---

# Must implement manually

These equations:

Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V

---

# Then stack

```python
GGNN → SelfAttention
```

---

# Output of Day 4

Temporal module complete.

---

# Day 5 — Build User-App Bipartite Graph

### One-line goal

Create personalized graph.

---

# Read

Personalized feature extraction module. 

---

# What graph looks like

```text
Users ↔ Apps
```

NOT app→app now.

---

# Example

```text
User1 → WhatsApp
User1 → YouTube
User2 → Gmail
```

---

# Build

```python
user_app_matrix
edge_index
node_features
```

---

# IMPORTANT

Paper uses:

```text
location features
```

You can initially ignore this.

Use learnable embeddings first.

Then add location later.

---

# Day 6 — Implement GCN

### One-line goal

Learn personalized node embeddings.

---

# Read

GCN equations carefully. 

---

# Implement

This exactly:

H^{(k)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(k-1)}W^{(k)}\right)

---

# Your target

```python
class GCNLayer(nn.Module)
```

---

# Understand deeply

GCN propagates:

```text
neighbor information
```

So users inherit app characteristics.

Apps inherit user preferences.

---

# Output of Day 6

User embeddings + App embeddings.

---

# Day 7 — Add Self Attention to Personalized Module

### One-line goal

Enhance personalized embeddings globally.

---

# Paper idea

GCN:

```text
local neighborhood
```

Self-attention:

```text
global dependency
```

Combined:

```text
multi-perspective representation learning
```

---

# Implement

```python
GCN → SelfAttention
```

Now personalized module complete.

---

# WEEK 2 — Fusion + Training

---

# Day 8 — Implement Feature Fusion

### One-line goal

Combine temporal + personalized embeddings.

---

# Read

Fusion equations. 

---

# Most important part

Attention-based fusion.

Implement:

\alpha_u = \frac{\exp(U_t^T W_u U_c)}{\exp(U_t^T W_u U_c)+\exp(U_c^T W_u U_t)}

Then:

U_f = \alpha_u U_t + (1-\alpha_u)U_c

---

# Output

Final embeddings:

```python
user_final
app_final
```

---

# Day 9 — Prediction Layer + Loss

### One-line goal

Predict next app.

---

# Read

Prediction section. 

---

# Implement

Score:

\hat{s}_i = softmax(U_f^T A_f)

Loss:

L = -\sum_{i=1}^{n}\left(s_i\log(\hat{s}_i)+(1-s_i)\log(1-\hat{s}_i)\right)+\gamma ||\theta||^2

---

# Day 10 — Training Pipeline

### One-line goal

Train end-to-end.

---

# Build

```python
train_loop()
evaluate()
save_model()
early_stopping()
```

---

# Paper hyperparameters

Use initially: 

```text
lr = 0.001
batch = 64
embedding = 64
epochs = 50
optimizer = Adam
```

---

# WEEK 3 — Evaluation + Debugging

---

# Day 11 — Metrics

### One-line goal

Implement evaluation metrics.

---

# Implement

```python
HR@K
MRR@K
NDCG@K
```

Paper uses:

```text
K = 1,5
```

---

# Day 12 — Reproduce Baselines

### One-line goal

Compare against simpler models.

---

# Start with

Implement only:

```text
GRU4Rec
SRGNN
```

Not all baselines initially.

---

# Day 13 — Ablation Studies

### One-line goal

Verify each module matters.

---

# Train:

* without self-attention
* without GCN
* without GGNN
* without fusion

Paper does same. 

---

# Day 14 — Final Cleanup

### One-line goal

Refactor into research-quality codebase.

---

# Final structure

```text
project/
│
├── data/
├── models/
│   ├── ggnn.py
│   ├── gcn.py
│   ├── attention.py
│   ├── fusion.py
│
├── trainer/
├── utils/
├── configs/
├── train.py
└── evaluate.py
```

---

# MOST IMPORTANT ADVICE

Do NOT start by implementing:

```text
full architecture
```

Start with:

```text
single module validation
```

Example:

```text
GGNN alone works?
GCN alone works?
Attention alone works?
```

Then combine.

This is how actual research engineers work.

---

# Another VERY Important Thing

The paper contains MANY research-level buzzwords.

But implementation-wise, it is mainly:

```text
GGNN
+
Transformer Attention
+
GCN
+
Attention Fusion
```

That’s it.

Do not feel overwhelmed by terminology.

---

# Final Implementation Order (CRITICAL)

Implement in THIS exact order:

```text
1. Dataset
2. Sessions
3. Directed Graph
4. GGNN
5. Self Attention
6. Bipartite Graph
7. GCN
8. Personalized Attention
9. Fusion
10. Prediction
11. Training
12. Metrics
13. Ablation
```

If you follow this order, implementation becomes manageable.

If you jump randomly, debugging becomes a nightmare.

---

# Strong Recommendation

Before coding each module:

1. Draw tensor shapes
2. Write tensor dimensions
3. Verify matrix multiplications

Example:

```text
Q = [batch, seq, d]
K = [batch, seq, d]
QK^T = [batch, seq, seq]
```

This alone prevents 70% of bugs.

---

# Your next best step

Start with:

```text
Day 1:
Implement dataset preprocessing and session construction only.
```

Do NOT touch neural networks yet.
