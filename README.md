# fake-news-detection-nlp
# Graph-Based Fake News Detection using PyTorch Geometric (GCN)

An applied machine learning pipeline implementing **Graph Convolutional Networks (GCN)** for document authenticity classification. Rather than analyzing articles in isolation, this architecture models topological text relationships by transforming articles into a semantic cosine-similarity graph over dense TF-IDF representations.

---

## 📌 Key Highlights
- **Graph Construction:** Vectorized cosine similarity thresholding ($> 0.3$) across 500-dimensional TF-IDF vectors, eliminating $O(N^2)$ pairwise loops.
- **Geometric Deep Learning:** 2-layer `GCNConv` network incorporating ReLU activations, dropout regularization ($p=0.5$), and transductive mask evaluation.
- **Evaluation:** Evaluated using multi-class Cross-Entropy Loss, classification accuracy, and macro F1-score.

---

## 🏗️ Architecture Pipeline

```text
Raw Text Corpus
      │
      ▼
Text Preprocessing (NLTK Tokenization, Stopword Removal, Regex Cleaning)
      │
      ▼
TF-IDF Feature Extraction (Sparse Vector Space)
      │
      ▼
Pairwise Cosine Similarity Thresholding (Adjacency Matrix Formulation)
      │
      ▼
PyTorch Geometric Data Object [Data(x=features, edge_index=edges, y=labels)]
      │
      ▼
Graph Convolutional Network (GCNConv -> ReLU -> Dropout -> GCNConv)
      │
      ▼
Transductive Semi-Supervised Node Classification

---

## 📊 Empirical Evaluation (Test Split: 1,000 Nodes)

| Metric | True News (Class 0) | Fake News (Class 1) | Macro Average |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.90 | 0.93 | **0.91** |
| **Recall** | 0.93 | 0.89 | **0.91** |
| **F1-Score** | 0.91 | 0.91 | **0.91** |
| **Accuracy** | — | — | **91.20%** |

### Training Dynamics
- **Optimization:** Adam optimizer ($lr=0.01$, $weight\_decay=5e-4$) with Cross-Entropy Loss.
- **Convergence:** Training loss dropped from `0.4161` (Epoch 10) to `0.2480` (Epoch 100).
- **Evaluation Split:** Evaluated on a transductive test mask of 1,000 unseen document nodes (496 True / 504 Fake) with dropout regularization ($p=0.5$).



