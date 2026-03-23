"""
NLP CIA2 - Full Sentence Ordering Pipeline
==========================================
Preprocessing : C  (preprocessing/preprocess.c)
Pipeline       : Python (pipeline/)

Ablation design
---------------
  Two ablation tables are printed:

  TABLE 1 — Pipeline Component Ablation
    Metric : Pairwise Ordering Accuracy (full pipeline end-to-end)
    For each component: WITH it active vs WITHOUT it (removed/replaced)

  TABLE 2 — Embedding x Multi-task Head Ablation   ← main focus
    For each embedding (TF-IDF, Word2Vec, BERT):
      For each of the 5 multi-task heads:
        WITH  = head trained using that embedding
        WITHOUT = head trained using RANDOM vectors (no semantic info)
        DELTA = WITH - WITHOUT  (shows what the embedding adds)
"""

import sys
import os
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

from dataset_generator import generate_dataset
from data_loader import compile_c, preprocess_docs, create_train_test_split
from semantic_stream import (
    TFIDFEncoder, Word2VecEncoder, BERTEncoder, get_sentence_corpus
)
from structural_stream import (
    GCNEncoder, build_local_graph, build_midrange_graph,
    build_global_graph, build_entity_graph, merge_graphs
)
from fusion import GatedFusion
from decoding import run_decoding
from multitask_heads import run_heads_with_encoder, run_multitask_heads

# ── Helpers ────────────────────────────────────────────────────────────────────

def print_banner(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def make_docs_with_sents(orig_docs, proc_results):
    out = []
    for orig, proc in zip(orig_docs, proc_results):
        sents = proc['preprocessed_sentences']
        if not sents or len(sents) != len(orig['sentences']):
            sents = orig['sentences']
        out.append({**orig, 'sentences': sents})
    return out


def build_pairs(docs, embed_fn):
    X, y = [], []
    for doc in docs:
        embs = embed_fn(doc)
        if embs is None or len(embs) < 2:
            continue
        n = len(embs)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                feat = np.concatenate([embs[i], embs[j],
                                       np.abs(embs[i]-embs[j]),
                                       embs[i]*embs[j]])
                X.append(feat)
                y.append(1 if i < j else 0)
    return np.array(X), np.array(y)


def lr_pairwise_acc(train_docs, test_docs, embed_fn):
    X_tr, y_tr = build_pairs(train_docs, embed_fn)
    X_te, y_te = build_pairs(test_docs,  embed_fn)
    if len(X_tr) == 0 or len(X_te) == 0:
        return 0.5
    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


def gnn_fn(tfidf_enc, gcn):
    def fn(doc):
        n    = len(doc['sentences'])
        init = tfidf_enc.encode_doc(doc)
        A = merge_graphs(build_local_graph(n), build_midrange_graph(n),
                         build_global_graph(init), build_entity_graph(doc['sentences']))
        return gcn.encode(A, init)
    return fn


def fused_fn(sem_enc, tfidf_enc, gcn, fusion):
    def fn(doc):
        sem    = sem_enc.encode_doc(doc)
        n      = len(doc['sentences'])
        init   = tfidf_enc.encode_doc(doc)
        A = merge_graphs(build_local_graph(n), build_midrange_graph(n),
                         build_global_graph(init), build_entity_graph(doc['sentences']))
        struct = gcn.encode(A, init)
        return fusion.fuse(sem, struct)
    return fn


def full_pipeline_acc(train_docs, test_docs, bert_enc,
                      preproc_flags=None, use_raw=False,
                      encoder_name='BERT', use_gnn=True, use_fusion=True):
    """Run complete pipeline for one variant, return final pairwise accuracy."""
    if preproc_flags is None:
        preproc_flags = []
    if use_raw:
        tr, te = list(train_docs), list(test_docs)
    else:
        tr = make_docs_with_sents(train_docs, preprocess_docs(train_docs, preproc_flags))
        te = make_docs_with_sents(test_docs,  preprocess_docs(test_docs,  preproc_flags))

    corpus = get_sentence_corpus(tr + te)
    if encoder_name == 'BERT':
        sem_enc = bert_enc
    elif encoder_name == 'TF-IDF':
        sem_enc = TFIDFEncoder(); sem_enc.fit(corpus)
    else:
        sem_enc = Word2VecEncoder(); sem_enc.fit(corpus)

    tfidf_gnn = TFIDFEncoder(); tfidf_gnn.fit(corpus)
    sample = tfidf_gnn.encode_doc(tr[0])
    gcn = GCNEncoder(input_dim=sample.shape[1], hidden_dim=128, output_dim=64)

    if use_gnn and use_fusion:
        all_sem = np.vstack([sem_enc.encode_doc(d) for d in tr])
        all_str = np.vstack([gnn_fn(tfidf_gnn, gcn)(d) for d in tr])
        out_dim = min(128, all_sem.shape[1])
        fus = GatedFusion(all_sem.shape[1], all_str.shape[1], output_dim=out_dim)
        fus.train_gate(all_sem, all_str, labels=None, n_iters=20)
        embed_fn = fused_fn(sem_enc, tfidf_gnn, gcn, fus)
    elif use_gnn:
        embed_fn = gnn_fn(tfidf_gnn, gcn)
    else:
        embed_fn = sem_enc.encode_doc

    return lr_pairwise_acc(tr, te, embed_fn)


# ── Random encoder (no semantic information — "WITHOUT" baseline) ──────────────

class RandomEncoder:
    """
    Returns fixed random Gaussian vectors per sentence.
    Represents a system with ZERO semantic knowledge.
    Any accuracy above this is purely due to the real embedding.
    """
    def __init__(self, dim=100, seed=42):
        self.dim  = dim
        self.seed = seed

    def fit(self, corpus=None):
        pass

    def encode_doc(self, doc):
        n   = len(doc['sentences'])
        rng = np.random.RandomState(self.seed + doc.get('id', 0))
        return rng.randn(n, self.dim)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print_banner("NLP CIA2 -- Sentence Ordering Pipeline")

    # ── 1. Dataset ────────────────────────────────────────────────
    print_banner("Step 1: Dataset Generation")
    docs = generate_dataset(seed=42)
    train_docs, test_docs = create_train_test_split(docs, test_ratio=0.2, seed=42)
    print(f"  {len(docs)} docs | Train: {len(train_docs)} | Test: {len(test_docs)}")
    print(f"  Topics: {sorted(set(d['topic'] for d in docs))}")

    # ── 2. C Preprocessor ─────────────────────────────────────────
    print_banner("Step 2: C Preprocessor")
    compile_c()
    train_proc = preprocess_docs(train_docs)
    test_proc  = preprocess_docs(test_docs)
    train_full = make_docs_with_sents(train_docs, train_proc)
    test_full  = make_docs_with_sents(test_docs,  test_proc)
    seg_acc = sum(
        1 for o, p in zip(test_docs, test_proc)
        if p['stats'].get('num_sentences', 0) == len(o['sentences'])
    ) / len(test_docs)
    print(f"  Sentence segmentation accuracy: {seg_acc:.2%}")

    # ── 3. Fit all encoders ───────────────────────────────────────
    print_banner("Step 3: Fitting Encoders")
    corpus = get_sentence_corpus(train_full + test_full)

    tfidf_enc = TFIDFEncoder();   tfidf_enc.fit(corpus);  print("  [1/3] TF-IDF fitted")
    w2v_enc   = Word2VecEncoder(); w2v_enc.fit(corpus);   print("  [2/3] Word2Vec fitted")
    bert_enc  = BERTEncoder();    bert_enc.fit();          print("  [3/3] BERT loaded")
    rand_enc  = RandomEncoder(dim=100, seed=42)            # "WITHOUT" baseline

    # ── 4. Build main GCN + Fusion (for full pipeline ablation) ───
    sample     = tfidf_enc.encode_doc(train_full[0])
    gcn_main   = GCNEncoder(input_dim=sample.shape[1], hidden_dim=128, output_dim=64)
    all_sem_tr = np.vstack([bert_enc.encode_doc(d) for d in train_full])
    all_str_tr = np.vstack([gnn_fn(tfidf_enc, gcn_main)(d) for d in train_full])
    fusion_main = GatedFusion(all_sem_tr.shape[1], all_str_tr.shape[1], output_dim=128)
    fusion_main.train_gate(all_sem_tr, all_str_tr, labels=None, n_iters=30)
    print("  GCN + Gated Fusion ready.")

    # ══════════════════════════════════════════════════════════════
    # TABLE 1 — Pipeline Component Ablation
    # Metric: Pairwise Ordering Accuracy (full pipeline, LR classifier)
    # ══════════════════════════════════════════════════════════════
    print_banner("TABLE 1: Pipeline Component Ablation")
    print(f"  Metric : Pairwise Ordering Accuracy (full pipeline end-to-end)")
    print(f"  Classifier : Logistic Regression on fused BERT+GNN embeddings")
    print(f"  WITH  = component active | WITHOUT = component removed\n")

    t1_rows = []

    print("  Computing baseline (all components ON)...")
    baseline = full_pipeline_acc(train_docs, test_docs, bert_enc)
    print(f"    Baseline: {baseline:.2%}\n")

    print("  [Preprocessing steps]")
    for step, wo_flags, use_raw in [
        ('Tokenization',       [],                 True),   # WITHOUT = raw text
        ('Lowercasing',        ['--no-lowercase'], False),
        ('Punct Removal',      ['--no-punct'],     False),
        ('Sent Segmentation',  ['--no-segment'],   False),
    ]:
        acc_wo = full_pipeline_acc(train_docs, test_docs, bert_enc,
                                    preproc_flags=wo_flags, use_raw=use_raw)
        t1_rows.append((step, 'preprocessing', baseline, acc_wo))
        d = baseline - acc_wo
        print(f"    {step:<22} WITH={baseline:.2%}  WITHOUT={acc_wo:.2%}  delta={d:>+.2%}")

    print("\n  [Semantic encoders]")
    for enc_name in ['TF-IDF', 'Word2Vec', 'BERT']:
        acc = full_pipeline_acc(train_docs, test_docs, bert_enc, encoder_name=enc_name)
        worst = min(
            full_pipeline_acc(train_docs, test_docs, bert_enc, encoder_name='TF-IDF'),
            full_pipeline_acc(train_docs, test_docs, bert_enc, encoder_name='Word2Vec'),
        ) if enc_name == 'BERT' else \
            full_pipeline_acc(train_docs, test_docs, bert_enc, encoder_name='Word2Vec') \
            if enc_name == 'TF-IDF' else \
            full_pipeline_acc(train_docs, test_docs, bert_enc, encoder_name='TF-IDF')
        t1_rows.append((f'Encoder: {enc_name}', 'vs weakest alt.', acc, worst))
        print(f"    {enc_name:<22} Accuracy={acc:.2%}")

    print("\n  [Structural + Fusion]")
    acc_no_gnn  = full_pipeline_acc(train_docs, test_docs, bert_enc,
                                     use_gnn=False, use_fusion=False)
    acc_no_fuse = full_pipeline_acc(train_docs, test_docs, bert_enc,
                                     use_gnn=True, use_fusion=False)
    t1_rows.append(('GNN Structural Stream', 'BERT only',    baseline, acc_no_gnn))
    t1_rows.append(('Gated Fusion',          'GNN emb only', baseline, acc_no_fuse))
    print(f"    {'With GNN+Fusion':<22} {baseline:.2%}  Without GNN={acc_no_gnn:.2%}  delta={baseline-acc_no_gnn:>+.2%}")
    print(f"    {'With Fusion':<22} {baseline:.2%}  Without Fuse={acc_no_fuse:.2%}  delta={baseline-acc_no_fuse:>+.2%}")

    # Print Table 1
    W, N = 26, 10
    print(f"\n  {'Component':<{W}} {'Note':<22} {'WITH':>{N}} {'WITHOUT':>{N}} {'DELTA':>{N}}")
    print("  " + "-" * (W + 22 + N * 3 + 4))
    for comp, note, acc_w, acc_wo in t1_rows:
        d = acc_w - acc_wo
        s = "+" if d >= 0 else ""
        print(f"  {comp:<{W}} {note:<22} {acc_w:>{N}.2%} {acc_wo:>{N}.2%} {s}{d:>{N-1}.2%}")

    # ══════════════════════════════════════════════════════════════
    # TABLE 2 — Embedding × Multi-task Head Ablation  (main focus)
    # ══════════════════════════════════════════════════════════════
    print_banner("TABLE 2: Embedding x Multi-task Head Ablation  [MAIN FOCUS]")
    print(f"  Metric : Per-head task accuracy")
    print(f"  WITH   : Head trained using this embedding")
    print(f"  WITHOUT: Head trained using RANDOM vectors (dim=100, no semantic info)")
    print(f"           Any accuracy above random = contribution of the embedding")
    print(f"  DELTA  : WITH - WITHOUT  (how much the embedding helps each head)\n")

    HEAD_NAMES = [
        'Sentence Ordering',
        'Pairwise Consistency',
        'Topic Continuity',
        'Entity Coherence',
        'Discourse Classification',
    ]

    EMBEDDINGS = [
        ('TF-IDF',   tfidf_enc),
        ('Word2Vec', w2v_enc),
        ('BERT',     bert_enc),
    ]

    # Compute "WITHOUT" baseline once (random encoder, same for all embeddings)
    print("  Computing WITHOUT baseline (random encoder)...")
    rand_results = run_heads_with_encoder(train_full, test_full, rand_enc)
    print(f"  Done.\n")

    # Compute WITH results for each embedding
    all_emb_results = {}
    for enc_name, enc in EMBEDDINGS:
        print(f"  Computing WITH {enc_name}...")
        all_emb_results[enc_name] = run_heads_with_encoder(train_full, test_full, enc)
        print(f"  Done.")

    # ── Print Table 2 ─────────────────────────────────────────────
    print()
    COL = 14

    # Header
    header = f"  {'Multi-task Head':<26}"
    for enc_name, _ in EMBEDDINGS:
        header += f" | {enc_name:^{COL*3+4}}"
    print(header)

    sub = f"  {'':<26}"
    for _ in EMBEDDINGS:
        sub += f" | {'WITH':>{COL}} {'W/OUT':>{COL}} {'DELTA':>{COL}}"
    print(sub)
    print("  " + "-" * (26 + (COL * 3 + 6) * len(EMBEDDINGS)))

    for head in HEAD_NAMES:
        row = f"  {head:<26}"
        rand_acc = rand_results[head]
        for enc_name, _ in EMBEDDINGS:
            with_acc = all_emb_results[enc_name][head]
            delta    = with_acc - rand_acc
            sign     = "+" if delta >= 0 else ""
            row += f" | {with_acc:>{COL}.2%} {rand_acc:>{COL}.2%} {sign+f'{delta:.2%}':>{COL}}"
        print(row)

    print("  " + "-" * (26 + (COL * 3 + 6) * len(EMBEDDINGS)))

    # Summary: best embedding per head
    print(f"\n  Best embedding per head:")
    for head in HEAD_NAMES:
        best_enc = max(EMBEDDINGS, key=lambda e: all_emb_results[e[0]][head])
        best_acc = all_emb_results[best_enc[0]][head]
        rand_acc = rand_results[head]
        print(f"    {head:<28}  {best_enc[0]:<10} {best_acc:.2%}  "
              f"(+{best_acc-rand_acc:.2%} over random)")

    # ── Full decoding final metrics ────────────────────────────────
    print_banner("Step 5: Full Decoding  (MLP scorer + tournament ranking)")
    decoding_results, _ = run_decoding(
        train_full, test_full, fusion_main, bert_enc, gcn_main, tfidf_enc
    )

    # ── Final summary ──────────────────────────────────────────────
    print_banner("FINAL SUMMARY")
    print(f"\n  Full Pipeline Output Metrics  (BERT + GNN + Gated Fusion + MLP):")
    print(f"  {'Metric':<45} {'Value':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Pairwise Ordering Accuracy':<45} {decoding_results['pairwise_accuracy']:>9.2%}")
    print(f"  {'Sequence Accuracy  (exact correct order)':<45} {decoding_results['sequence_accuracy']:>9.2%}")
    print(f"  {'Kendall Tau        (ordering correlation)':<45} {decoding_results['kendall_tau']:>9.4f}")

    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete.")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
