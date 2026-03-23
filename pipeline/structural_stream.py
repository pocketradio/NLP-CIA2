"""
structural_stream.py
Graph-based structural stream: four graph types, GCN encoder,
and pairwise sentence ordering evaluation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')


# ── Similarity helper ─────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Four graph types ──────────────────────────────────────────────────────────

def build_local_graph(n):
    """Adjacent-sentence edges: (i, i+1) with weight 1."""
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i][i + 1] = 1.0
        A[i + 1][i] = 1.0
    return A


def build_midrange_graph(n, window=3):
    """Edges between sentences within a sliding window."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, min(i + window + 1, n)):
            A[i][j] = 1.0
            A[j][i] = 1.0
    return A


def build_global_graph(embeddings, threshold=0.3):
    """Cosine-similarity edges above threshold."""
    n = len(embeddings)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                A[i][j] = sim
                A[j][i] = sim
    return A


def build_entity_graph(sentences):
    """
    Shared-entity edges.
    Entities are approximated as: capitalised words or words longer than 4 characters.
    Edge weight = Jaccard overlap of entity sets.
    """
    n = len(sentences)
    A = np.zeros((n, n))

    def get_entities(sent):
        words = sent.split()
        return set(
            w.lower().strip('.,!?;:"\'-')
            for w in words
            if len(w) > 4 or (len(w) > 1 and w[0].isupper())
        )

    entities = [get_entities(s) for s in sentences]
    for i in range(n):
        for j in range(i + 1, n):
            union = entities[i] | entities[j]
            if not union:
                continue
            overlap = len(entities[i] & entities[j])
            if overlap > 0:
                score = overlap / len(union)
                A[i][j] = score
                A[j][i] = score
    return A


def merge_graphs(local, midrange, global_g, entity,
                 weights=(0.4, 0.25, 0.2, 0.15)):
    """Weighted combination of the four adjacency matrices."""
    return (weights[0] * local
            + weights[1] * midrange
            + weights[2] * global_g
            + weights[3] * entity)


# ── Graph Construction Accuracy ───────────────────────────────────────────────

def graph_construction_accuracy(docs, tfidf_encoder):
    """
    Measure how well the local graph captures true adjacency.
    Correct = local edge exists for truly adjacent pair, no edge for non-adjacent.
    Returns accuracy over all (i,j) pairs in all docs.
    """
    correct = 0
    total   = 0
    for doc in docs:
        n = len(doc['sentences'])
        if n < 2:
            continue
        A_local = build_local_graph(n)
        # Adjacent pairs should have edge
        for i in range(n - 1):
            total   += 1
            correct += int(A_local[i][i + 1] > 0)
        # Non-adjacent pairs should NOT have edge in local graph
        for i in range(n):
            for j in range(i + 2, n):
                total   += 1
                correct += int(A_local[i][j] == 0)
    return correct / max(total, 1)


# ── GCN (numpy) ───────────────────────────────────────────────────────────────

def gcn_layer(A, X, W, b, activation='relu'):
    """
    Single GCN propagation layer.
    Computes: H = act( D^{-1/2} A_hat D^{-1/2} X W + b )
    where A_hat = A + I (self-loops added).
    """
    n = A.shape[0]
    A_hat = A + np.eye(n)
    degree = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(degree) + 1e-8))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    Z = A_norm @ X @ W + b
    if activation == 'relu':
        return np.maximum(0.0, Z)
    elif activation == 'tanh':
        return np.tanh(Z)
    return Z


class GCNEncoder:
    """Two-layer Graph Convolutional Network for structural encoding."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=64, seed=42):
        np.random.seed(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale2
        self.b2 = np.zeros(output_dim)

    def encode(self, A, X):
        """
        A : (n, n) combined adjacency matrix
        X : (n, input_dim) initial node features
        Returns: (n, output_dim) structural node embeddings
        """
        H1 = gcn_layer(A, X, self.W1, self.b1, activation='relu')
        H2 = gcn_layer(A, H1, self.W2, self.b2, activation='tanh')
        return H2


def get_initial_features(doc, tfidf_encoder):
    """Use TF-IDF embeddings as initial node features."""
    return tfidf_encoder.encode_doc(doc)


# ── Structural Dataset ────────────────────────────────────────────────────────

def build_struct_dataset(docs, gcn, tfidf_encoder):
    """Build pairwise dataset using GCN-derived structural embeddings."""
    X, y = [], []
    for doc in docs:
        sents = doc['sentences']
        n = len(sents)
        if n < 2:
            continue

        init_feats = get_initial_features(doc, tfidf_encoder)

        A_local    = build_local_graph(n)
        A_mid      = build_midrange_graph(n)
        A_global   = build_global_graph(init_feats)
        A_entity   = build_entity_graph(sents)
        A_combined = merge_graphs(A_local, A_mid, A_global, A_entity)

        struct_embs = gcn.encode(A_combined, init_feats)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = np.abs(struct_embs[i] - struct_embs[j])
                prod = struct_embs[i] * struct_embs[j]
                feat = np.concatenate([struct_embs[i], struct_embs[j], diff, prod])
                X.append(feat)
                y.append(1 if i < j else 0)

    if not X:
        return np.zeros((0, 1)), np.zeros(0, dtype=int)
    return np.array(X), np.array(y)


# ── Run Structural Stream ─────────────────────────────────────────────────────

def run_structural_stream(train_docs, test_docs, tfidf_encoder):
    """
    Run GCN structural stream.
    Returns:
        results    – dict with 'graph_construction' and 'gnn_pairwise' accuracies
        gcn        – fitted GCNEncoder instance
        clf        – fitted LogisticRegression on structural features
    """
    print("  Building GCN encoder...")

    # Determine input dimension from first document
    sample_feats = tfidf_encoder.encode_doc(train_docs[0])
    input_dim    = sample_feats.shape[1]
    output_dim   = 64

    gcn = GCNEncoder(input_dim=input_dim, hidden_dim=128, output_dim=output_dim)

    # Graph construction accuracy (based on local graph correctness)
    graph_acc = graph_construction_accuracy(train_docs, tfidf_encoder)
    print(f"  Graph Construction Accuracy: {graph_acc:.4f}")

    # Build pairwise datasets
    X_train, y_train = build_struct_dataset(train_docs, gcn, tfidf_encoder)
    X_test,  y_test  = build_struct_dataset(test_docs,  gcn, tfidf_encoder)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    gnn_acc = clf.score(X_test, y_test)
    print(f"  GNN Structural Pairwise Accuracy: {gnn_acc:.4f}")

    return {
        'graph_construction': graph_acc,
        'gnn_pairwise':       gnn_acc,
    }, gcn, clf
