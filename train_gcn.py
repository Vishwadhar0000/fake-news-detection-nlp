
# STEP 1: Data Loading and Preprocessing
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the dataset
fake = pd.read_csv('fake updated.csv')
true = pd.read_csv('true updated.csv')

# Label the datasets
fake['label'] = 1
true['label'] = 0

# Combine and shuffle
df = pd.concat([fake[['title', 'text', 'label']], true[['title', 'text', 'label']]])
df.dropna(inplace=True)

# Text Cleaning Function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = word_tokenize(text.lower())
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# STEP 2: TF-IDF Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(df['clean_text'])

# STEP 3: Graph Construction using Cosine Similarity
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(X_tfidf)
threshold = 0.3
G = nx.Graph()

for i in range(len(df)):
    G.add_node(i, label=df.iloc[i]['label'])

for i in range(len(df)):
    for j in range(i+1, len(df)):
        if cos_sim[i][j] > threshold:
            G.add_edge(i, j, weight=cos_sim[i][j])

# STEP 4: Convert to PyTorch Geometric Format
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

features = torch.tensor(X_tfidf.toarray(), dtype=torch.float)
labels = torch.tensor(df['label'].values, dtype=torch.long)

for i, node in enumerate(G.nodes()):
    G.nodes[node]['x'] = features[i]
    G.nodes[node]['y'] = labels[i]

data = from_networkx(G)
data.x = features
data.y = labels

# STEP 5: Define the GCN Model
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# STEP 6: Training the GCN
from sklearn.metrics import accuracy_score
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=features.shape[1], hidden_dim=64, output_dim=2).to(device)
data = data.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Create masks
train_mask = torch.rand(len(labels)) < 0.8
test_mask = ~train_mask

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        pred = out[test_mask].argmax(dim=1)
        acc = accuracy_score(data.y[test_mask].cpu(), pred.cpu())
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')
