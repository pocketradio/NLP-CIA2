"""
fusion.py
Gated MLP fusion module combining semantic and structural embeddings.
Compares semantic-only, structural-only, and fused pairwise accuracies.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')


# ── Gated Fusion Module ───────────────────────────────────────────────────────

class GatedFusion:
    """
    Gated MLP fusion.
    Both semantic and structural streams are projected to `output_dim`.
    A sigmoid gate (learned via random search) blends the two projections:
        fused = gate * p_sem + (1 - gate) * p_struct
    """

    def __init__(self, sem_dim, struct_dim, output_dim=128, seed=42):
        np.random.seed(seed)
        self.output_dim = output_dim

        # Projection matrices
        self.W_sem    = np.random.randn(sem_dim,    output_dim) * np.sqrt(2.0 / sem_dim)
        self.W_struct = np.random.randn(struct_dim, output_dim) * np.sqrt(2.0 / struct_dim)

        # Gate weights
        self.W_gate = np.random.randn(output_dim * 2, output_dim) * 0.01
        self.b_gate = np.zeros(output_dim)
        self.trained = False

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def project(self, sem_emb, struct_emb):
        """Project both streams to output_dim."""
        p_sem    = np.tanh(sem_emb    @ self.W_sem)
        p_struct = np.tanh(struct_emb @ self.W_struct)
        return p_sem, p_struct

    def fuse(self, sem_emb, struct_emb):
        """
        Fuse semantic and structural embeddings via gated combination.
        Inputs can be 1-D (single sentence) or 2-D (batch of sentences).
        """
        p_sem, p_struct = self.project(sem_emb, struct_emb)
        combined = np.concatenate([p_sem, p_struct], axis=-1)
        gate     = self.sigmoid(combined @ self.W_gate + self.b_gate)
        return gate * p_sem + (1.0 - gate) * p_struct

    def train_gate(self, sem_train, struct_train, labels=None,
                   n_iters=50, lr=0.01):
        """
        Train gate weights using random search (perturbation + keep-if-better).
        sem_train   : (N, sem_dim)
        struct_train: (N, struct_dim)
        """
        best_loss = float('inf')
        best_W    = self.W_gate.copy()
        best_b    = self.b_gate.copy()

        np.random.seed(42)
        for _ in range(n_iters):
            noise_W = np.random.randn(*self.W_gate.shape) * 0.1
            noise_b = np.random.randn(*self.b_gate.shape) * 0.1
            self.W_gate += noise_W
            self.b_gate += noise_b

            fused  = self.fuse(sem_train, struct_train)
            target = np.tanh((sem_train @ self.W_sem + struct_train @ self.W_struct) / 2.0)
            loss   = float(np.mean((fused - target) ** 2))

            if loss < best_loss:
                best_loss = loss
                best_W    = self.W_gate.copy()
                best_b    = self.b_gate.copy()
            else:
                self.W_gate = best_W.copy()
                self.b_gate = best_b.copy()

        self.trained = True


# ── Embedding helpers ─────────────────────────────────────────────────────────

def get_doc_embeddings(docs, semantic_encoder, gcn, tfidf_encoder):
    """Compute both semantic and structural embeddings for every doc."""
    from structural_stream import (
        build_local_graph, build_midrange_graph,
        build_global_graph, build_entity_graph, merge_graphs,
    )

    doc_embs = []
    for doc in docs:
        sents = doc['sentences']
        n     = len(sents)

        sem_embs   = semantic_encoder.encode_doc(doc)
        init_feats = tfidf_encoder.encode_doc(doc)

        A_local    = build_local_graph(n)
        A_mid      = build_midrange_graph(n)
        A_global   = build_global_graph(init_feats)
        A_entity   = build_entity_graph(sents)
        A_combined = merge_graphs(A_local, A_mid, A_global, A_entity)

        struct_embs = gcn.encode(A_combined, init_feats)

        doc_embs.append({
            'sem':    sem_embs,
            'struct': struct_embs,
            'n':      n,
        })
    return doc_embs


# ── Pairwise dataset from embeddings ─────────────────────────────────────────

def _build_pairwise_from_embs(embs_fn, doc_embs, docs):
    """
    Generic helper: apply `embs_fn(doc_emb) -> (n, d)` to get embeddings,
    then build pairwise feature vectors.
    Returns (X list, y list).
    """
    X, y = [], []
    for e, d in zip(doc_embs, docs):
        embs = embs_fn(e)
        n    = e['n']
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                feat = np.concatenate([
                    embs[i], embs[j],
                    np.abs(embs[i] - embs[j]),
                    embs[i] * embs[j],
                ])
                X.append(feat)
                y.append(1 if i < j else 0)
    return X, y


def _pairwise_accuracy(get_feat_fn, train_embs, test_embs, train_docs, test_docs):
    """Train LogReg on pairwise features and return test accuracy."""
    X_tr, y_tr = _build_pairwise_from_embs(get_feat_fn, train_embs, train_docs)
    X_te, y_te = _build_pairwise_from_embs(get_feat_fn, test_embs,  test_docs)

    if not X_tr or not X_te:
        return 0.5

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(np.array(X_tr), np.array(y_tr))
    return float(clf.score(np.array(X_te), np.array(y_te)))


# ── Run Fusion ────────────────────────────────────────────────────────────────

def run_fusion(train_docs, test_docs, semantic_encoder, gcn, tfidf_encoder):
    """
    Run fusion module and compare semantic-only, structural-only, and fused.
    Returns:
        results       – dict {semantic_only, structural_only, fusion}
        fusion_module – trained GatedFusion instance
    """
    print("  Computing embeddings for fusion...")

    train_embs = get_doc_embeddings(train_docs, semantic_encoder, gcn, tfidf_encoder)
    test_embs  = get_doc_embeddings(test_docs,  semantic_encoder, gcn, tfidf_encoder)

    sem_dim    = train_embs[0]['sem'].shape[1]
    struct_dim = train_embs[0]['struct'].shape[1]
    output_dim = min(128, sem_dim)

    fusion = GatedFusion(sem_dim, struct_dim, output_dim)

    # Train gate on all training embeddings
    all_sem    = np.vstack([e['sem']    for e in train_embs])
    all_struct = np.vstack([e['struct'] for e in train_embs])
    fusion.train_gate(all_sem, all_struct, n_iters=30)

    # Feature extraction lambdas
    def sem_fn(e):
        return e['sem']

    def struct_fn(e):
        return e['struct']

    def fused_fn(e):
        return fusion.fuse(e['sem'], e['struct'])

    results = {}
    results['semantic_only']   = _pairwise_accuracy(sem_fn,    train_embs, test_embs, train_docs, test_docs)
    results['structural_only'] = _pairwise_accuracy(struct_fn, train_embs, test_embs, train_docs, test_docs)
    results['fusion']          = _pairwise_accuracy(fused_fn,  train_embs, test_embs, train_docs, test_docs)

    print(f"  Semantic only:   {results['semantic_only']:.4f}")
    print(f"  Structural only: {results['structural_only']:.4f}")
    print(f"  Fusion:          {results['fusion']:.4f}")

    return results, fusion
