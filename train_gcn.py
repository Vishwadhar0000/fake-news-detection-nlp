import re
import nltk
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.optim as optim

# Ensure required NLTK corpora are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- STEP 1: Data Loading & Preprocessing ---
def load_and_preprocess(fake_path='fake updated.csv', true_path='true updated.csv'):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake['label'] = 1
    true['label'] = 0

    df = pd.concat([fake[['title', 'text', 'label']], true[['title', 'text', 'label']]]).dropna()
    
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
        tokens = word_tokenize(text)
        return ' '.join([w for w in tokens if w not in stop_words])

    df['clean_text'] = df['text'].apply(clean_text)
    return df

# --- STEP 2: TF-IDF & Vectorized Graph Construction ---
def build_graph(df, threshold=0.3, max_features=500):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(df['clean_text']).toarray()

    # Calculate dense pairwise cosine similarity
    cos_sim = cosine_similarity(X_tfidf)
    
    # Vectorized edge creation (replaces slow O(N^2) loop)
    adj_matrix = cos_sim > threshold
    np.fill_diagonal(adj_matrix, False)
    
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
    features = torch.tensor(X_tfidf, dtype=torch.float)
    labels = torch.tensor(df['label'].values, dtype=torch.long)

    return Data(x=features, edge_index=edge_index, y=labels)

# --- STEP 3: GCN Architecture ---
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- STEP 4: Training Pipeline ---
def train_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")

    df = load_and_preprocess()
    data = build_graph(df).to(device)

    # Semi-supervised transductive masks (80% train / 20% test)
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True

    model = GCN(input_dim=data.num_node_features, hidden_dim=64, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = out[test_mask].argmax(dim=1)
                acc = accuracy_score(data.y[test_mask].cpu(), pred.cpu())
                f1 = f1_score(data.y[test_mask].cpu(), pred.cpu(), average='macro')
                print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Test Acc: {acc:.4f} | Test Macro-F1: {f1:.4f}")

    # Final Classification Summary
    model.eval()
    with torch.no_grad():
        final_preds = out[test_mask].argmax(dim=1).cpu()
        print("\nFinal Test Set Performance:")
        print(classification_report(data.y[test_mask].cpu(), final_preds, target_names=['True News', 'Fake News']))

if __name__ == "__main__":
    train_pipeline()
