```text
======================================================================
  NLP CIA2 -- Sentence Ordering Pipeline
======================================================================

======================================================================
  Step 1: ACL Anthology Dataset Loading
======================================================================
  [Loader] Reading abstract.csv...
  [Loader] Reached limit of 100 documents.
  [Loader] Successfully loaded 100 documents.
  100 ACL abstracts loaded | Train: 80 | Test: 20

======================================================================
  Step 2: C Preprocessor
======================================================================
  [INFO] gcc not found. C source exists at preprocessing/preprocess.c
         Using Python reimplementation of the same preprocessing logic.
         To use the actual C binary: install MinGW (choco install mingw)
         or Scoop (scoop install gcc), then re-run.
  Sentence segmentation accuracy: 100.00%

======================================================================
  Step 3: Fitting Encoders
======================================================================
  [1/5] TF-IDF fitted
  [2/5] Word2Vec fitted
  Loading BERT model: all-MiniLM-L6-v2 ...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 2708.06it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  BERT model loaded.
  [3/5] SBERT loaded
  [4/5] Word2Vec+TFIDF fitted
  Loading Transformer model: distilbert-base-uncased ...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 4342.69it/s]
DistilBertModel LOAD REPORT from: distilbert-base-uncased
Key                     | Status     |  |
------------------------+------------+--+-
vocab_transform.bias    | UNEXPECTED |  |
vocab_layer_norm.weight | UNEXPECTED |  |
vocab_projector.bias    | UNEXPECTED |  |
vocab_transform.weight  | UNEXPECTED |  |
vocab_layer_norm.bias   | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Transformer model distilbert-base-uncased loaded.
  [5/6] Raw DistilBERT loaded
  Preparing Siamese pairwise dataset for distilbert-base-uncased...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 5374.76it/s]
DistilBertModel LOAD REPORT from: distilbert-base-uncased
Key                     | Status     |  |
------------------------+------------+--+-
vocab_transform.bias    | UNEXPECTED |  |
vocab_layer_norm.weight | UNEXPECTED |  |
vocab_projector.bias    | UNEXPECTED |  |
vocab_transform.weight  | UNEXPECTED |  |
vocab_layer_norm.bias   | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  Fine-tuning distilbert-base-uncased (Siamese Margin Ranking)...
Training Siamese Network (Epoch 1/3): 100%|███████████████████████████████████████████████████████████| 106/106 [07:02<00:00,  3.99s/it] 
  Epoch 1/3 complete. Avg Loss: 0.4408
Training Siamese Network (Epoch 2/3): 100%|███████████████████████████████████████████████████████████| 106/106 [04:34<00:00,  2.59s/it] 
  Epoch 2/3 complete. Avg Loss: 0.1040
Training Siamese Network (Epoch 3/3): 100%|███████████████████████████████████████████████████████████| 106/106 [04:41<00:00,  2.65s/it] 
  Epoch 3/3 complete. Avg Loss: 0.0711
  [6/6] Fine-Tuned DistilBERT fitted
  Evaluating sequence classification head performance for Fine-Tuned DistilBERT:
  Evaluating Siamese scalar scorer on test set...
  Siamese Scorer Pairwise Accuracy: 0.7484

======================================================================
  Step 4: Embedding Evaluation (Pairwise Accuracy in Final Step)
======================================================================

  Evaluating Embedding: 1. Word2Vec (Mean Pooled)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.5220
  Final Result (1. Word2Vec (Mean Pooled)) -> Pairwise Ordering Accuracy: 52.20%

  Evaluating Embedding: 2. Word2Vec (TF-IDF Weighted)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.4780
  Final Result (2. Word2Vec (TF-IDF Weighted)) -> Pairwise Ordering Accuracy: 47.80%

  Evaluating Embedding: 3. TF-IDF (Sparse Vector)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.4843
  Final Result (3. TF-IDF (Sparse Vector)) -> Pairwise Ordering Accuracy: 48.43%

  Evaluating Embedding: 4. Contextual Token (DistilBERT)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.6698
  Final Result (4. Contextual Token (DistilBERT)) -> Pairwise Ordering Accuracy: 66.98%

  Evaluating Embedding: 5. Sequence Domain (SBERT)
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.6447
  Final Result (5. Sequence Domain (SBERT)) -> Pairwise Ordering Accuracy: 64.47%

  Evaluating Embedding: 6. Fine-Tuned Token (DistilBERT [CLS])
  Computing fused embeddings...
  Training pairwise scorer (MLP)...
  Pairwise Scoring Accuracy: 0.7547
  Final Result (6. Fine-Tuned Token (DistilBERT [CLS])) -> Pairwise Ordering Accuracy: 75.47%

======================================================================
  Pipeline complete.
======================================================================
```
