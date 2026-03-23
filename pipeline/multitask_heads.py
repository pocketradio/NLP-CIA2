"""
multitask_heads.py
Five auxiliary multi-task training heads built on top of fused embeddings:
  1. Sentence Ordering   – predict first-half vs second-half position
  2. Pairwise Consistency – pairwise ordering + transitivity check
  3. Topic Continuity     – adjacent vs non-adjacent sentence relatedness
  4. Entity Coherence     – consecutive sentence entity overlap
  5. Discourse Classification – intro / body / conclusion role
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')


# ── Embedding helper ──────────────────────────────────────────────────────────

def get_all_embeddings(docs, semantic_encoder, gcn, tfidf_encoder, fusion_module):
    """Compute fused (+ semantic) embeddings for all docs."""
    from structural_stream import (
        build_local_graph, build_midrange_graph,
        build_global_graph, build_entity_graph, merge_graphs,
    )

    all_embs = []
    for doc in docs:
        sents = doc['sentences']
        n     = len(sents)

        sem    = semantic_encoder.encode_doc(doc)
        init   = tfidf_encoder.encode_doc(doc)

        A = merge_graphs(
            build_local_graph(n),
            build_midrange_graph(n),
            build_global_graph(init),
            build_entity_graph(sents),
        )
        struct = gcn.encode(A, init)
        fused  = fusion_module.fuse(sem, struct)

        all_embs.append({
            'fused': fused,
            'sem':   sem,
            'sents': sents,
            'n':     n,
        })
    return all_embs


# ── Head 1: Sentence Ordering ─────────────────────────────────────────────────

def head_sentence_ordering(train_embs, test_embs):
    """
    Binary classification: first-half (label=0) vs second-half (label=1) position.
    Uses fused embedding of each sentence as feature.
    """
    def build(embs_list):
        X, y = [], []
        for e in embs_list:
            n = e['n']
            for pos, emb in enumerate(e['fused']):
                X.append(emb)
                y.append(0 if pos < n // 2 else 1)
        return np.array(X), np.array(y)

    X_tr, y_tr = build(train_embs)
    X_te, y_te = build(test_embs)

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


# ── Head 2: Pairwise Consistency ──────────────────────────────────────────────

def head_pairwise_consistency(train_embs, test_embs):
    """
    Train a pairwise classifier, then measure transitivity consistency:
    fraction of (i,j,k) triples where if pred[i,j]=1 and pred[j,k]=1
    then pred[i,k]=1.
    """
    def build_pairwise(embs_list):
        X, y = [], []
        for e in embs_list:
            fused = e['fused']
            n     = e['n']
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    feat = np.concatenate([
                        fused[i], fused[j],
                        np.abs(fused[i] - fused[j]),
                    ])
                    X.append(feat)
                    y.append(1 if i < j else 0)
        return np.array(X), np.array(y)

    X_tr, y_tr = build_pairwise(train_embs)
    X_te, y_te = build_pairwise(test_embs)

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    # Measure transitivity on test set predictions
    idx = 0
    consistent    = 0
    total_triples = 0

    for e in test_embs:
        n     = e['n']
        pairs = n * (n - 1)

        # Rebuild predicted pairwise matrix from flat predictions
        pred_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    if idx < len(preds):
                        pred_matrix[i][j] = preds[idx]
                        idx += 1

        # Check transitivity: for all (i,j,k)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    total_triples += 1
                    if pred_matrix[i][j] == 1 and pred_matrix[j][k] == 1:
                        if pred_matrix[i][k] == 1:
                            consistent += 1
                        # else: transitivity violated – do NOT increment
                    else:
                        consistent += 1  # constraint not triggered, counts as fine

    return consistent / max(total_triples, 1)


# ── Head 3: Topic Continuity ──────────────────────────────────────────────────

def head_topic_continuity(train_embs, test_embs):
    """
    Binary: adjacent sentence pair = topically related (1),
    non-adjacent pair with cosine sim < 0.5 = not related (0).
    Uses semantic embeddings as features.
    """
    def cosine_sim(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def build(embs_list):
        X, y = [], []
        for e in embs_list:
            sem = e['sem']
            n   = e['n']
            # Adjacent pairs → related
            for i in range(n - 1):
                feat = np.concatenate([
                    sem[i], sem[i + 1],
                    np.abs(sem[i] - sem[i + 1]),
                    sem[i] * sem[i + 1],
                ])
                X.append(feat)
                y.append(1)
            # Non-adjacent pairs
            for i in range(n):
                for j in range(n):
                    if abs(i - j) > 1:
                        feat = np.concatenate([
                            sem[i], sem[j],
                            np.abs(sem[i] - sem[j]),
                            sem[i] * sem[j],
                        ])
                        sim = cosine_sim(sem[i], sem[j])
                        X.append(feat)
                        y.append(1 if sim > 0.5 else 0)
        return np.array(X), np.array(y)

    X_tr, y_tr = build(train_embs)
    X_te, y_te = build(test_embs)

    if len(np.unique(y_tr)) < 2:
        return float(np.mean(y_te == y_tr[0]))

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


# ── Head 4: Entity Coherence ──────────────────────────────────────────────────

def head_entity_coherence(train_embs, test_embs):
    """
    Binary: predict whether two consecutive sentences share named entities.
    Entity proxy: capitalised words or words with length > 4.
    Features: fused embeddings of the two sentences.
    """
    def get_entities(sent):
        words = sent.split()
        return set(
            w.lower().strip('.,!?;:\'"')
            for w in words
            if len(w) > 4 or (len(w) > 1 and w[0].isupper())
        )

    def build(embs_list):
        X, y = [], []
        for e in embs_list:
            fused = e['fused']
            sents = e['sents']
            n     = e['n']
            for i in range(n - 1):
                ents_i = get_entities(sents[i])
                ents_j = get_entities(sents[i + 1])
                overlap = len(ents_i & ents_j)
                label   = 1 if overlap > 0 else 0
                feat    = np.concatenate([
                    fused[i], fused[i + 1],
                    np.abs(fused[i] - fused[i + 1]),
                ])
                X.append(feat)
                y.append(label)
        return np.array(X), np.array(y)

    X_tr, y_tr = build(train_embs)
    X_te, y_te = build(test_embs)

    if len(X_tr) == 0 or len(X_te) == 0:
        return 0.5

    if len(np.unique(y_tr)) < 2:
        # All same class – return majority baseline
        majority = y_tr[0]
        return float(np.mean(y_te == majority))

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


# ── Head 5: Discourse Classification ─────────────────────────────────────────

def head_discourse_classification(train_embs, test_embs):
    """
    3-class: intro (pos=0) → 0, body (pos=1..n-2) → 1, conclusion (pos=n-1) → 2.
    Features: fused embedding of each sentence.
    """
    def build(embs_list):
        X, y = [], []
        for e in embs_list:
            fused = e['fused']
            n     = e['n']
            for pos, emb in enumerate(fused):
                if pos == 0:
                    label = 0          # intro
                elif pos == n - 1:
                    label = 2          # conclusion
                else:
                    label = 1          # body
                X.append(emb)
                y.append(label)
        return np.array(X), np.array(y)

    X_tr, y_tr = build(train_embs)
    X_te, y_te = build(test_embs)

    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


# ── Run All Heads (fused pipeline) ────────────────────────────────────────────

def run_multitask_heads(train_docs, test_docs, semantic_encoder,
                        gcn, tfidf_encoder, fusion_module):
    """
    Run all five heads using fused (BERT+GNN) embeddings.
    Returns dict of {head_name: accuracy}.
    """
    print("  Computing embeddings for multi-task heads...")
    train_embs = get_all_embeddings(
        train_docs, semantic_encoder, gcn, tfidf_encoder, fusion_module
    )
    test_embs = get_all_embeddings(
        test_docs, semantic_encoder, gcn, tfidf_encoder, fusion_module
    )

    results = {}
    print("  [Head 1] Sentence Ordering...")
    results['sentence_ordering'] = head_sentence_ordering(train_embs, test_embs)
    print("  [Head 2] Pairwise Consistency...")
    results['pairwise_consistency'] = head_pairwise_consistency(train_embs, test_embs)
    print("  [Head 3] Topic Continuity...")
    results['topic_continuity'] = head_topic_continuity(train_embs, test_embs)
    print("  [Head 4] Entity Coherence...")
    results['entity_coherence'] = head_entity_coherence(train_embs, test_embs)
    print("  [Head 5] Discourse Classification...")
    results['discourse_classification'] = head_discourse_classification(
        train_embs, test_embs
    )
    for k, v in results.items():
        print(f"    {k}: {v:.4f}")
    return results


# ── Run All Heads with a given encoder directly ────────────────────────────────

def run_heads_with_encoder(train_docs, test_docs, encoder):
    """
    Run all five heads using a single encoder's output as both
    'fused' and 'sem' embeddings.  Used for embedding ablation:
    plug in TF-IDF / Word2Vec / BERT / Random and compare results.

    Parameters
    ----------
    encoder : object with .encode_doc(doc) -> np.ndarray (n_sents, dim)

    Returns
    -------
    dict {head_name: accuracy}
    """
    def make_embs(docs):
        out = []
        for doc in docs:
            embs = encoder.encode_doc(doc)
            out.append({
                'fused': embs,          # used by heads 1, 2, 4, 5
                'sem':   embs,          # used by head 3 (topic continuity)
                'sents': doc['sentences'],
                'n':     len(doc['sentences']),
            })
        return out

    tr = make_embs(train_docs)
    te = make_embs(test_docs)

    return {
        'Sentence Ordering':        head_sentence_ordering(tr, te),
        'Pairwise Consistency':     head_pairwise_consistency(tr, te),
        'Topic Continuity':         head_topic_continuity(tr, te),
        'Entity Coherence':         head_entity_coherence(tr, te),
        'Discourse Classification': head_discourse_classification(tr, te),
    }
