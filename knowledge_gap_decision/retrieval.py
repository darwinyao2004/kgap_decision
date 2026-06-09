import math
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


@dataclass
class RetrievalScores:
    top1_similarity: float
    top5_avg_similarity: float
    ranked_indices: list[int]


def tfidf_scores(query: str, documents: list[str]) -> RetrievalScores:
    docs = [d for d in documents if d]
    if not docs:
        return RetrievalScores(0.0, 0.0, [])
    try:
        vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
        matrix = vectorizer.fit_transform([query] + docs)
        sims = (matrix[0] @ matrix[1:].T).toarray().ravel()
    except ValueError:
        sims = np.zeros(len(docs))
    order = np.argsort(-sims).tolist()
    top1 = float(sims[order[0]]) if order else 0.0
    top5 = float(np.mean([sims[i] for i in order[:5]])) if order else 0.0
    return RetrievalScores(top1, top5, order)


def bm25_like_score(query: str, documents: list[str]) -> RetrievalScores:
    q = tokenize(query)
    docs_tokens = [tokenize(d) for d in documents if d]
    if not q or not docs_tokens:
        return RetrievalScores(0.0, 0.0, [])
    df = {term: sum(term in set(doc) for doc in docs_tokens) for term in set(q)}
    n = len(docs_tokens)
    avgdl = sum(len(doc) for doc in docs_tokens) / max(n, 1)
    scores = []
    for doc in docs_tokens:
        score = 0.0
        counts = {t: doc.count(t) for t in set(doc)}
        for term in q:
            if term not in counts:
                continue
            idf = math.log((n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            tf = counts[term]
            score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * len(doc) / max(avgdl, 1)))
        scores.append(score)
    arr = np.array(scores, dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max()
    order = np.argsort(-arr).tolist()
    return RetrievalScores(float(arr[order[0]]), float(np.mean(arr[order[:5]])), order)
