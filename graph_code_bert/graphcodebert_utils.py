import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json

class GraphCodeBERTRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.linear(x)

def build_dependency_graph_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    prev = None
    for entry in data:
        ident = entry["ident"]
        G.add_node(ident)
        if prev and ident != prev:
            G.add_edge(prev, ident)
        prev = ident

    labels = {n: len(nx.descendants(G, n)) for n in G.nodes()}
    return labels

def get_graphcodebert_embeddings(labels_dict, model_name="microsoft/graphcodebert-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    dataset = []
    for ident, score in labels_dict.items():
        tokens = tokenizer(ident, return_tensors="pt")
        with torch.no_grad():
            output = model(**tokens)
        emb = output.last_hidden_state[:, 0, :].squeeze()  # CLS token
        dataset.append((emb, score))
    return dataset

def train_regression_model(dataset, epochs=100, lr=1e-3):
    X = torch.stack([x for x, y in dataset])
    y = torch.tensor([y for x, y in dataset], dtype=torch.float).unsqueeze(1)
    model = GraphCodeBERTRegressor(X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model, X

def predict_and_rank(model, X, labels_dict):
    with torch.no_grad():
        pred = model(X).squeeze()
        sorted_result = sorted(zip(labels_dict.keys(), pred.tolist()), key=lambda x: -x[1])
    print("\n\U0001F4CA 예측된 전파량 순위:")
    for name, score in sorted_result:
        print(f"{name:<10} | Predicted Influence: {score:.3f}")