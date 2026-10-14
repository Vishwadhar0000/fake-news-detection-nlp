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
