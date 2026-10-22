"""
Module 4: Consistency, Coherence, and Semantic Drift Engine
Compiler Analogy: Semantic Verification, Type Checking, and Coherence Analysis

Responsibilities:
1. Baseline Scorer: Word-set Jaccard similarity (|A ∩ B| / |A ∪ B|).
2. Dense Embedding Scorers:
   - spaCy vector cosine similarity (pooled content words).
   - Sentence-Transformers cosine similarity (all-MiniLM-L6-v2) for contextual semantic representation.
3. Proposed Hybrid Scorer:
   Hybrid Score = W_TRIPLET * Soft_Triplet_Sim + W_VECTOR * Embedding_Sim
   Default weights: W_TRIPLET = 0.4, W_VECTOR = 0.6 (tunable).
   TRIPLET_MATCH_THRESHOLD = 0.60 (named constant).
4. Contradiction Detection Engine:
   Identifies conflicting propositions across paragraph pairs sharing identical/synonymous
   subject and object (soft similarity >= 0.80) but possessing opposite polarity predicates
   ('reduce' vs 'not reduce').
5. Drift Threshold Calibration:
   Evaluates precision, recall, and F1 across thresholds (0.25 - 0.50) on labelled pair benchmarks.
6. Unified Anomaly Flag Engine:
   Returns both consecutive semantic drift logs and cross-paragraph logical contradictions.
7. Visualizations:
   Seaborn consistency heatmaps and NetworkX Knowledge Graph (saved at 300 dpi).
"""

from itertools import combinations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import spacy
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .module1_lexical import get_spacy_model, lexical_analysis, get_content_words
from .module2_triplets import extract_triplets

# Named constants as specified in project specifications
W_TRIPLET: float = 0.4
W_VECTOR: float = 0.6
DRIFT_THRESHOLD: float = 0.35
TRIPLET_MATCH_THRESHOLD: float = 0.60
CONTRADICTION_ENTITY_THRESHOLD: float = 0.80

# Lazy-loaded SentenceTransformer model
_sbert_model = None

def get_sentence_transformer():
    """Lazily load and cache all-MiniLM-L6-v2 sentence transformer model."""
    global _sbert_model
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Notice: SentenceTransformer could not be loaded ({e}). Falling back to spaCy vectors.")
            _sbert_model = None
    return _sbert_model


def compute_jaccard_similarity(doc1: spacy.tokens.Doc, doc2: spacy.tokens.Doc) -> float:
    """
    Baseline Scorer: Word-set Jaccard similarity on content words.
    Jaccard = |Set(A) ∩ Set(B)| / |Set(A) ∪ Set(B)|
    """
    words1 = set(get_content_words(doc1))
    words2 = set(get_content_words(doc2))
    union = words1.union(words2)
    if not union:
        return 0.0
    return float(len(words1.intersection(words2)) / len(union))


def _token_span_similarity(nlp, span1: str, span2: str) -> float:
    """Compute dense cosine similarity between two entity or predicate spans."""
    d1 = nlp(span1)
    d2 = nlp(span2)
    if d1.vector_norm and d2.vector_norm:
        return float(d1.similarity(d2))
    return 0.0


def compute_triplet_similarity(
    triplets1: List[Tuple[str, str, str]],
    triplets2: List[Tuple[str, str, str]],
    match_threshold: float = TRIPLET_MATCH_THRESHOLD,
    nlp=None
) -> float:
    """
    Soft Jaccard similarity across triplet sets using dense vector embeddings.
    Two triplets match if the average cosine similarity of Subject, Predicate,
    and Object is >= match_threshold (default 0.60).
    """
    if not triplets1 or not triplets2:
        return 0.0

    if nlp is None:
        nlp = get_spacy_model()

    # Match triplets in set 1 against set 2
    matched_1 = 0
    for s1, p1, o1 in triplets1:
        best_sim = 0.0
        for s2, p2, o2 in triplets2:
            s_sim = _token_span_similarity(nlp, s1, s2)
            p_sim = _token_span_similarity(nlp, p1, p2)
            o_sim = _token_span_similarity(nlp, o1, o2)
            avg_sim = (s_sim + p_sim + o_sim) / 3.0
            if avg_sim > best_sim:
                best_sim = avg_sim
        if best_sim >= match_threshold:
            matched_1 += 1

    # Match triplets in set 2 against set 1
    matched_2 = 0
    for s2, p2, o2 in triplets2:
        best_sim = 0.0
        for s1, p1, o1 in triplets1:
            s_sim = _token_span_similarity(nlp, s1, s2)
            p_sim = _token_span_similarity(nlp, p1, p2)
            o_sim = _token_span_similarity(nlp, o1, o2)
            avg_sim = (s_sim + p_sim + o_sim) / 3.0
            if avg_sim > best_sim:
                best_sim = avg_sim
        if best_sim >= match_threshold:
            matched_2 += 1

    avg_matched = (matched_1 + matched_2) / 2.0
    denominator = (len(triplets1) + len(triplets2)) - avg_matched
    return float(avg_matched / denominator) if denominator > 0 else 0.0


def compute_spacy_vector_similarity(
    doc1: spacy.tokens.Doc,
    doc2: spacy.tokens.Doc,
    nlp=None
) -> float:
    """
    Dense vector cosine similarity using spaCy en_core_web_md on content words.
    """
    if nlp is None:
        nlp = get_spacy_model()

    words1 = " ".join(get_content_words(doc1))
    words2 = " ".join(get_content_words(doc2))

    d1 = nlp(words1)
    d2 = nlp(words2)

    if d1.vector_norm and d2.vector_norm:
        sim = float(d1.similarity(d2))
        return float(np.clip(sim, 0.0, 1.0))
    return 0.0


def compute_sbert_similarity(text1: str, text2: str) -> float:
    """
    Dense sentence-transformer cosine similarity using all-MiniLM-L6-v2.
    Falls back to spaCy vectors if sentence-transformers is unavailable.
    """
    sbert = get_sentence_transformer()
    if sbert is None:
        nlp = get_spacy_model()
        return compute_spacy_vector_similarity(nlp(text1), nlp(text2), nlp=nlp)

    emb = sbert.encode([text1, text2], convert_to_numpy=True)
    norm1 = np.linalg.norm(emb[0])
    norm2 = np.linalg.norm(emb[1])
    if norm1 > 0 and norm2 > 0:
        sim = float(np.dot(emb[0], emb[1]) / (norm1 * norm2))
        return float(np.clip(sim, 0.0, 1.0))
    return 0.0


def compute_vector_similarity(
    doc1: spacy.tokens.Doc,
    doc2: spacy.tokens.Doc,
    use_sbert: bool = False,
    nlp=None
) -> float:
    """
    Unified vector similarity interface supporting both spaCy and Sentence-Transformers.
    """
    if use_sbert:
        return compute_sbert_similarity(doc1.text, doc2.text)
    return compute_spacy_vector_similarity(doc1, doc2, nlp=nlp)


def compute_hybrid_score(
    doc1: spacy.tokens.Doc,
    doc2: spacy.tokens.Doc,
    w_triplet: float = W_TRIPLET,
    w_vector: float = W_VECTOR,
    use_sbert: bool = False,
    nlp=None
) -> Tuple[float, float, float]:
    """
    Compute proposed hybrid coherence score:
    Hybrid = w_triplet * Triplet_Sim + w_vector * Vector_Sim
    Returns (hybrid_score, triplet_sim, vector_sim).
    """
    if nlp is None:
        nlp = get_spacy_model()

    t1 = extract_triplets(doc1)
    t2 = extract_triplets(doc2)

    t_sim = compute_triplet_similarity(t1, t2, match_threshold=TRIPLET_MATCH_THRESHOLD, nlp=nlp)
    v_sim = compute_vector_similarity(doc1, doc2, use_sbert=use_sbert, nlp=nlp)
    hybrid = w_triplet * t_sim + w_vector * v_sim

    return float(round(hybrid, 3)), float(round(t_sim, 3)), float(round(v_sim, 3))


def detect_contradictions(
    triplets_per_paragraph: List[List[Tuple[str, str, str]]],
    entity_threshold: float = CONTRADICTION_ENTITY_THRESHOLD,
    nlp=None
) -> List[Dict[str, Any]]:
    """
    Detect logical contradictions across paragraph pairs:
    Finds pairs of triplets sharing Subject and Object (similarity >= entity_threshold)
    where one predicate is the negation of the other (e.g. 'reduce' vs 'not reduce').
    """
    if nlp is None:
        nlp = get_spacy_model()

    all_triplets = []
    for pid, plist in enumerate(triplets_per_paragraph, start=1):
        for s, p, o in plist:
            all_triplets.append((pid, s, p, o))

    contradictions = []
    for (p1, s1, pred1, o1), (p2, s2, pred2, o2) in combinations(all_triplets, 2):
        if p1 == p2:
            continue

        # Check Subject match
        s_match = (s1.lower() == s2.lower()) or (
            _token_span_similarity(nlp, s1, s2) >= entity_threshold
        )
        if not s_match:
            continue

        # Check Object match
        o_match = (o1.lower() == o2.lower()) or (
            _token_span_similarity(nlp, o1, o2) >= entity_threshold
        )
        if not o_match:
            continue

        # Check Polarity / Negation mismatch
        neg1 = pred1.startswith("not ")
        neg2 = pred2.startswith("not ")

        if neg1 != neg2:
            base1 = pred1[4:].strip() if neg1 else pred1.strip()
            base2 = pred2[4:].strip() if neg2 else pred2.strip()

            verb_match = (base1 == base2) or (
                _token_span_similarity(nlp, base1, base2) >= 0.70
            )

            if verb_match:
                contradictions.append({
                    "paragraph_a": p1,
                    "paragraph_b": p2,
                    "triplet_a": (s1, pred1, o1),
                    "triplet_b": (s2, pred2, o2),
                    "subject": s1,
                    "object": o1,
                    "predicate_a": pred1,
                    "predicate_b": pred2,
                    "finding": f"Logical Contradiction: P{p1} states '{pred1}' vs P{p2} states '{pred2}' on ({s1}, {o1})"
                })

    return contradictions


def build_coherence_matrix(
    paragraphs: List[str],
    w_triplet: float = W_TRIPLET,
    w_vector: float = W_VECTOR,
    use_sbert: bool = False,
    nlp=None
) -> Tuple[np.ndarray, List[spacy.tokens.Doc], List[list]]:
    """
    Build symmetric N x N coherence matrix for a list of paragraphs.
    Diagonal is 1.0. Off-diagonals are hybrid scores.
    """
    if nlp is None:
        nlp = get_spacy_model()

    docs = [lexical_analysis(p, nlp=nlp)[0] for p in paragraphs]
    triplets_list = [extract_triplets(d) for d in docs]

    n = len(docs)
    matrix = np.eye(n)

    for i, j in combinations(range(n), 2):
        t_sim = compute_triplet_similarity(triplets_list[i], triplets_list[j], match_threshold=TRIPLET_MATCH_THRESHOLD, nlp=nlp)
        v_sim = compute_vector_similarity(docs[i], docs[j], use_sbert=use_sbert, nlp=nlp)
        score = round(w_triplet * t_sim + w_vector * v_sim, 3)
        matrix[i, j] = matrix[j, i] = score

    return matrix, docs, triplets_list


def detect_semantic_drift(matrix: np.ndarray, threshold: float = DRIFT_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Scan consecutive paragraphs (P_i -> P_{i+1}) and flag semantic drift
    whenever the coherence score drops below threshold.
    """
    n = len(matrix)
    drift_events = []
    for i in range(n - 1):
        score = matrix[i, i + 1]
        is_drift = bool(score < threshold)
        event = {
            "from_paragraph": i + 1,
            "to_paragraph": i + 2,
            "score": float(score),
            "is_drift": is_drift,
            "log_message": f"WARNING: semantic drift between P{i+1} and P{i+2} (score {score:.2f})"
            if is_drift
            else f"OK: coherent transition between P{i+1} and P{i+2} (score {score:.2f})"
        }
        drift_events.append(event)
    return drift_events


def flag_document_anomalies(
    paragraphs: List[str],
    w_triplet: float = W_TRIPLET,
    w_vector: float = W_VECTOR,
    drift_threshold: float = DRIFT_THRESHOLD,
    use_sbert: bool = False,
    nlp=None
) -> Dict[str, Any]:
    """
    Unified anomaly detection engine returning:
    - Coherence matrix
    - Semantic drift log (consecutive paragraph drop < threshold)
    - Contradiction log (cross-paragraph logical assertions that clash)
    """
    if nlp is None:
        nlp = get_spacy_model()

    matrix, docs, triplets_list = build_coherence_matrix(
        paragraphs, w_triplet=w_triplet, w_vector=w_vector, use_sbert=use_sbert, nlp=nlp
    )
    drift_log = detect_semantic_drift(matrix, threshold=drift_threshold)
    contradictions = detect_contradictions(triplets_list, nlp=nlp)

    return {
        "matrix": matrix,
        "docs": docs,
        "triplets": triplets_list,
        "drift_events": drift_log,
        "contradictions": contradictions,
        "drift_count": sum(1 for d in drift_log if d["is_drift"]),
        "contradiction_count": len(contradictions)
    }


def calibrate_drift_threshold(
    labelled_pairs_df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    nlp=None
) -> Tuple[float, pd.DataFrame]:
    """
    Perform threshold calibration across labelled pair benchmark.
    Tests candidate thresholds from 0.25 to 0.50 in steps of 0.05.
    Returns (best_threshold, calibration_results_dataframe).
    """
    if thresholds is None:
        thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    if nlp is None:
        nlp = get_spacy_model()

    # Precompute hybrid scores for all pairs
    scores = []
    ground_truth_drift = []

    for _, row in labelled_pairs_df.iterrows():
        d1 = nlp(row["text_a"])
        d2 = nlp(row["text_b"])
        hyb, _, _ = compute_hybrid_score(d1, d2, nlp=nlp)
        scores.append(hyb)
        # Ground truth: drifting is positive class (1 = drift, 0 = consistent)
        is_drift = 1 if row["label"] == "drifting" else 0
        ground_truth_drift.append(is_drift)

    scores = np.array(scores)
    y_true = np.array(ground_truth_drift)

    results = []
    best_f1 = -1.0
    best_th = 0.35

    for th in thresholds:
        # Predict drift if score < threshold
        y_pred = (scores < th).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

        results.append({
            "Threshold": round(th, 2),
            "Accuracy": round(accuracy, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1_Score": round(f1, 3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        })

    calib_df = pd.DataFrame(results)
    return best_th, calib_df


def plot_calibration_curve(
    calib_df: pd.DataFrame,
    best_th: float,
    output_path: str = "outputs/module4_calibration_curve.png",
    ax=None
) -> None:
    """Plot F1, Precision, and Recall curve across drift thresholds. Saves at 300 dpi."""
    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        save_fig = True

    ax.plot(calib_df["Threshold"], calib_df["F1_Score"], marker="o", linewidth=2.2, color="#2b8a3e", label="F1 Score")
    ax.plot(calib_df["Threshold"], calib_df["Precision"], marker="s", linewidth=1.8, linestyle="--", color="#1c7ed6", label="Precision")
    ax.plot(calib_df["Threshold"], calib_df["Recall"], marker="^", linewidth=1.8, linestyle=":", color="#f59f00", label="Recall")

    ax.axvline(best_th, color="#e03131", linestyle="-.", alpha=0.8, label=f"Calibrated Threshold ({best_th:.2f})")
    ax.axvline(DRIFT_THRESHOLD, color="#868e96", linestyle=":", alpha=0.8, label=f"Default Threshold ({DRIFT_THRESHOLD:.2f})")

    ax.set_title("Module 4: Drift Threshold Calibration on Labelled Corpus", fontsize=11, fontweight="bold")
    ax.set_xlabel("Semantic Drift Threshold", fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True)

    if save_fig:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_heatmap(
    matrix: np.ndarray,
    output_path: str = "outputs/heatmap.png",
    title: str = "Paragraph Consistency Matrix (Coherence Heatmap)",
    ax=None
) -> None:
    """
    Render color-coded N x N paragraph consistency heatmap using Seaborn.
    Values >= 0.35 green/yellow (coherent), < 0.35 red (semantic drift).
    Saves figure at 300 dpi.
    """
    labels = [f"P{i+1}" for i in range(len(matrix))]
    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        save_fig = True

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Hybrid Coherence Score"}
    )
    ax.set_title(title, fontsize=11, fontweight="bold")

    if save_fig:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_knowledge_graph(
    triplets_list: List[List[Tuple[str, str, str]]],
    output_path: str = "outputs/knowledge_graph.png",
    title: str = "Compiler-Derived Entity-Relationship Knowledge Graph",
    ax=None
) -> None:
    """
    Construct directed Knowledge Graph (AST semantic network) from extracted triplets.
    Nodes represent entities; directed edges represent predicate relationships.
    Saves figure at 300 dpi.
    """
    G = nx.DiGraph()
    for plist in triplets_list:
        for s, p, o in plist:
            G.add_edge(s, o, label=p)

    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 7.5))
        save_fig = True

    pos = nx.spring_layout(G, k=0.9, seed=42)
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color="#cfe3ff",
        edge_color="#6c757d",
        node_size=2400,
        font_size=8,
        font_weight="bold",
        arrows=True,
        arrowsize=15
    )
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")

    if save_fig:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
