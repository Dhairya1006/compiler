"""
Coherence Engine - Unified Package Exports
Compiler-Inspired Multi-Paragraph Document Verification Pipeline
"""

from .module1_lexical import (
    get_spacy_model,
    load_paragraphs,
    clean_text,
    lexical_analysis,
    get_content_words
)

from .module2_triplets import (
    span_text,
    get_subjects,
    get_objects,
    extract_triplets
)

from .module3_symbol_table import (
    PRONOUNS,
    SymbolTable,
    resolve_triplets_with_symbol_table,
    compute_entity_drift,
    plot_entity_drift,
    compare_with_fastcoref
)

from .module4_scoring import (
    W_TRIPLET,
    W_VECTOR,
    DRIFT_THRESHOLD,
    TRIPLET_MATCH_THRESHOLD,
    CONTRADICTION_ENTITY_THRESHOLD,
    get_sentence_transformer,
    compute_jaccard_similarity,
    compute_triplet_similarity,
    compute_spacy_vector_similarity,
    compute_sbert_similarity,
    compute_vector_similarity,
    compute_hybrid_score,
    detect_contradictions,
    build_coherence_matrix,
    detect_semantic_drift,
    flag_document_anomalies,
    calibrate_drift_threshold,
    plot_calibration_curve,
    plot_heatmap,
    plot_knowledge_graph
)
