#  Sentiment Analysis on Amazon Food Reviews  
Group 39 — Suchitra Hole, Bhakti Pasnani, Vaishnavi Bhutada  

##  Project Overview

This project explores **sentiment classification** of food product reviews from Amazon using various NLP techniques. The goal is to classify each review into one of three sentiment classes: **Positive**, **Neutral**, or **Negative**, by leveraging both traditional and deep learning models with different types of text embeddings.

We compare multiple pipelines combining:
- Vectorizers: **TF-IDF**, **GloVe**, and **Sentence-BERT**
- Classifiers: **Logistic Regression**, **SVM**, and **CNN**

Our goal is to evaluate these combinations for performance and generalizability across key metrics like **Accuracy** and **F1 Score**.

---

##  Dataset

- **Source**: [Kaggle - Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- **Classes**:
  - Scores 1–2 → Negative
  - Score 3 → Neutral
  - Scores 4–5 → Positive

---

##  Preprocessing Steps

- Lowercasing text  
- Removing punctuation and non-alphanumeric characters  
- Removing stopwords using `NLTK` and `sklearn`  
- Tokenization using `nltk.word_tokenize` and `Keras Tokenizer`

---

##  Model Pipelines

###  Vectorization Techniques:
- **TF-IDF**: For traditional frequency-based representation  
- **GloVe**: Pre-trained embeddings (`glove.6B.100d.txt`)  
- **Sentence-BERT**: `all-MiniLM-L6-v2` via `sentence-transformers`

###  Model-Embedding Combinations:
- TF-IDF + Logistic Regression / SVM / CNN  
- GloVe + Logistic Regression / SVM / CNN  
- BERT + CNN

---

##  Evaluation

Metrics used:
- Accuracy  
- Weighted F1 Score  
- Confusion Matrix  
- Classification Report  

Used **k-Fold Cross Validation** to avoid overfitting and ensure model robustness.

---

##  Results Summary

| Model Combination     | Accuracy | Notes                                  |
|-----------------------|----------|----------------------------------------|
| GloVe + CNN           | **90%**  | Best overall performance               |
| TF-IDF + SVM          | ~86%     | Strong traditional baseline            |
| BERT + CNN            | ~88%     | Good contextual understanding          |

---

##  Key Learnings

- Embedding choice significantly affects performance  
- CNNs paired with semantic/contextual embeddings (GloVe/BERT) outperform traditional classifiers  
- Preprocessing consistency is essential for fair comparison  
- BERT shows promise even without fine-tuning  

---


