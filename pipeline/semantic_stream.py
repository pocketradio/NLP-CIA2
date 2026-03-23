"""
semantic_stream.py
Three sentence encoders (TF-IDF, Word2Vec, BERT) and pairwise classification
for semantic sentence ordering.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')


def get_sentence_corpus(docs):
    """Extract all sentences from a list of docs."""
    corpus = []
    for doc in docs:
        corpus.extend(doc['sentences'])
    return corpus


# ── TF-IDF Sentence Embeddings ────────────────────────────────────────────────

class TFIDFEncoder:
    """TF-IDF vectorizer wrapped as a sentence encoder."""

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.fitted = False

    def fit(self, corpus):
        """Fit TF-IDF vectorizer on corpus (list of strings)."""
        self.vec.fit(corpus)
        self.fitted = True

    def encode(self, sentences):
        """Returns (n_sents, d) numpy array of TF-IDF sentence embeddings."""
        return self.vec.transform(sentences).toarray()

    def encode_doc(self, doc):
        """Encode all sentences in a document dict."""
        return self.encode(doc['sentences'])


# ── Word2Vec Sentence Embeddings ──────────────────────────────────────────────

class Word2VecEncoder:
    """Mean-pooled Word2Vec sentence encoder."""

    def __init__(self, vector_size=100, window=5, min_count=1, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, corpus):
        """Train Word2Vec on tokenized corpus."""
        try:
            from gensim.models import Word2Vec
            tokenized = [sent.lower().split() for sent in corpus]
            self.model = Word2Vec(
                sentences=tokenized,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                epochs=self.epochs,
                workers=1,
                seed=42,
            )
        except ImportError:
            print("  [Warning] gensim not found. Using random Word2Vec embeddings.")
            self.model = None

    def _sentence_vector(self, sentence):
        """Mean-pool word vectors for a sentence."""
        if self.model is None:
            rng = np.random.default_rng(abs(hash(sentence)) % (2 ** 31))
            return rng.standard_normal(self.vector_size).astype(np.float32)
        words = sentence.lower().split()
        vecs = []
        for w in words:
            if w in self.model.wv:
                vecs.append(self.model.wv[w])
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def encode(self, sentences):
        """Returns (n_sents, vector_size) numpy array."""
        return np.array([self._sentence_vector(s) for s in sentences])

    def encode_doc(self, doc):
        """Encode all sentences in a document dict."""
        return self.encode(doc['sentences'])


# ── BERT Sentence Embeddings ──────────────────────────────────────────────────

class BERTEncoder:
    """Sentence-Transformers BERT encoder with TF-IDF fallback."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self._fallback_vec = None

    def fit(self, corpus=None):
        """Load pretrained sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading BERT model: {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)
            print("  BERT model loaded.")
        except ImportError:
            print("  [Warning] sentence-transformers not found. Using TF-IDF fallback for BERT.")
            self.model = None
            if corpus:
                self._fallback_vec = TfidfVectorizer(max_features=1000)
                self._fallback_vec.fit(corpus)

    def encode(self, sentences):
        """Returns (n_sents, embed_dim) numpy array."""
        if self.model is not None:
            return self.model.encode(sentences, show_progress_bar=False)
        # Fallback: TF-IDF + random projection to 384 dims
        if self._fallback_vec is not None:
            X = self._fallback_vec.transform(sentences).toarray()
        else:
            vec = TfidfVectorizer(max_features=1000)
            X = vec.fit_transform(sentences).toarray()
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((X.shape[1], 384)) / np.sqrt(384)
        return X @ proj

    def encode_doc(self, doc):
        """Encode all sentences in a document dict."""
        return self.encode(doc['sentences'])


# ── Pairwise Classifier Utilities ─────────────────────────────────────────────

def build_pairwise_features(emb_i, emb_j):
    """Build feature vector for a sentence pair (i, j)."""
    diff = np.abs(emb_i - emb_j)
    prod = emb_i * emb_j
    return np.concatenate([emb_i, emb_j, diff, prod])


def build_dataset(docs, encoder):
    """Build pairwise classification dataset from docs using given encoder."""
    X, y = [], []
    for doc in docs:
        embs = encoder.encode_doc(doc)
        n = len(doc['sentences'])
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                feat = build_pairwise_features(embs[i], embs[j])
                X.append(feat)
                y.append(1 if i < j else 0)
    if not X:
        return np.zeros((0, 1)), np.zeros(0, dtype=int)
    return np.array(X), np.array(y)


def evaluate_encoder(encoder, train_docs, test_docs, name):
    """Train a pairwise logistic-regression classifier and return accuracy + clf."""
    print(f"  Evaluating {name} encoder...")

    X_train, y_train = build_dataset(train_docs, encoder)
    X_test,  y_test  = build_dataset(test_docs,  encoder)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"    {name}: insufficient data, returning 0.5")
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        return 0.5, clf

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"    {name} pairwise accuracy: {acc:.4f}")
    return acc, clf


def run_semantic_stream(train_docs, test_docs):
    """
    Run all three semantic encoders.
    Returns:
        results  – dict {encoder_name: accuracy}
        encoders – dict {encoder_name: (encoder_obj, clf)}
    """
    corpus = get_sentence_corpus(train_docs + test_docs)

    results  = {}
    encoders = {}

    # ── TF-IDF ──
    tfidf = TFIDFEncoder()
    tfidf.fit(corpus)
    acc, clf = evaluate_encoder(tfidf, train_docs, test_docs, 'TF-IDF')
    results['TF-IDF']  = acc
    encoders['TF-IDF'] = (tfidf, clf)

    # ── Word2Vec ──
    w2v = Word2VecEncoder()
    w2v.fit(corpus)
    acc, clf = evaluate_encoder(w2v, train_docs, test_docs, 'Word2Vec')
    results['Word2Vec']  = acc
    encoders['Word2Vec'] = (w2v, clf)

    # ── BERT ──
    bert = BERTEncoder()
    bert.fit(corpus)
    acc, clf = evaluate_encoder(bert, train_docs, test_docs, 'BERT')
    results['BERT']  = acc
    encoders['BERT'] = (bert, clf)

    return results, encoders
