"""
Module 1: Ingestion and Lexical Analysis
Compiler Analogy: Lexer / Scanner Phase

Responsibilities:
1. Ingest .txt and .docx files.
2. Segment document into sequential paragraphs, dropping fragments (< 5 words).
3. Tokenize, remove stopwords, lemmatize, and generate Part-of-Speech (POS) tags.
4. Extract filtered content words (nouns, proper nouns, verbs, adjectives).
"""

import os
import re
import pandas as pd
import spacy

# Global spaCy medium pipeline with dense vectors
_nlp = None

def get_spacy_model():
    """Lazily load and return the en_core_web_md spaCy pipeline."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_md")
    return _nlp


def load_paragraphs(file_path: str) -> list[str]:
    """
    Ingest text from .txt or .docx file and split into paragraphs.
    Filters out empty lines and short headings (< 5 words).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.lower().endswith(".docx"):
        import docx
        doc = docx.Document(file_path)
        raw_paragraphs = [p.text for p in doc.paragraphs]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_paragraphs = f.read().split("\n")

    # Keep paragraphs with at least 5 whitespace-separated words
    clean_paras = [p.strip() for p in raw_paragraphs if len(p.strip().split()) >= 5]
    return clean_paras


def clean_text(text: str) -> str:
    """
    Lexical normalization: collapse multiple spaces and strip extraneous symbols
    while preserving grammatical punctuation needed for syntax parsing.
    """
    text = re.sub(r"\s+", " ", text)
    # Retain alphanumeric characters, standard punctuation, and hyphens
    text = re.sub(r"[^\w\s.,;:!?'\"-]", "", text)
    return text.strip()


def lexical_analysis(text: str, nlp=None) -> tuple[spacy.tokens.Doc, pd.DataFrame]:
    """
    Execute lexical pass on input text.
    Returns:
        doc: Processed spaCy Doc object.
        token_df: DataFrame containing token, POS, lemma, and is_stop fields.
    """
    if nlp is None:
        nlp = get_spacy_model()

    normalized = clean_text(text)
    doc = nlp(normalized)

    records = []
    for token in doc:
        records.append({
            "Token": token.text,
            "POS": token.pos_,
            "Lemma": token.lemma_.lower() if token.lower_ != "data" else "data",
            "Is_Stop": token.is_stop
        })

    token_df = pd.DataFrame(records)
    return doc, token_df


def get_content_words(doc: spacy.tokens.Doc) -> list[str]:
    """
    Extract normalized content words (nouns, proper nouns, verbs, adjectives).
    Strips stopwords, punctuation, and non-alphabetic tokens.
    Used for vocabulary overlap and content vector similarity in Module 4.
    """
    content = [
        t.lemma_.lower() if t.lower_ != "data" else "data"
        for t in doc
        if t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
        and not t.is_stop
        and t.is_alpha
    ]
    return content
