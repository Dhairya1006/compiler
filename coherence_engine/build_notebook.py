"""
Script to generate demo.ipynb with detailed markdown commentary,
compiler theory connections, and executable cells.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# Compiler-Guided Dependency Tree Triplet Extraction & Contextual Coherence Engine
**Subject:** Compiler Design / Natural Language Processing  
**Team No.:** 3  
**Team Members:** Raghav Arora (24BCE2299) & Dhairya Sawant (24BCE2319)  

---

### Executive Overview & Compiler Phase Mapping
In standard compiler architecture, source code undergoes lexical, syntactic, and semantic verification to ensure structural and typing validity. This project builds a direct structural correspondence to verify multi-paragraph natural language documents:

| Compiler Phase | Natural Language Equivalent | System Technical Implementation |
| :--- | :--- | :--- |
| **Lexical Analysis (Lexer)** | Token Stream & POS Tagging | Cleaning, tokenization, POS tagging (`NOUN`, `VERB`, `ADJ`), lemmatization, stopword filtering. |
| **Syntax Analysis (Parser / AST)** | Dependency Tree Parsing | Constructing dependency ASTs to extract Subject–Predicate–Object (SPO) triplets across active & passive clauses. |
| **Semantic Analysis & Symbol Table** | Coreference Tracking & Coherence Engine | Global Symbol Table for pronoun scope resolution + Hybrid Coherence (Soft-Jaccard Triplet Alignment + Dense Cosine Vector Similarity). |
| **Error / Diagnostic Logging** | Semantic Drift Warning Log | Flags consecutive paragraph transitions with similarity $< 0.35$ as context breaks. |
"""))

# Cell 1: Environment & Engine Imports
cells.append(nbf.v4.new_markdown_cell("""## Step 1: Environment Setup & Engine Import
Load the core compiler engine, language models, and verify vector embedding capabilities."""))

cells.append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import spacy
from spacy import displacy

# Import our custom compiler engine
from engine import (
    load_paragraphs,
    clean,
    lexical_pass,
    extract_triplets,
    SymbolTable,
    build_symbol_table,
    triplet_similarity,
    vector_similarity,
    analyse,
    plot_knowledge_graph,
    plot_heatmap,
    generate_baseline_comparison_table,
    detect_contradictions,
    compute_entity_drift,
    nlp,
    W_TRIPLET,
    W_VECTOR,
    DRIFT_THRESHOLD
)

print(f"spaCy version: {spacy.__version__}")
print(f"Model loaded: {nlp.meta['name']}")
print(f"Dense vector support verified: {nlp('Hello world').has_vector}")
"""))

# Cell 2: Document Ingestion
cells.append(nbf.v4.new_markdown_cell("""## Step 2: Document Ingestion & Paragraph Chunking (Input Layer)
The input layer reads multi-page documents (`.txt` or `.docx`), splits them into sequential paragraph nodes $[P_1, P_2, \dots, P_n]$, and filters out short noise headers."""))

cells.append(nbf.v4.new_code_cell("""# Ingest sample multi-paragraph benchmark document
data_path = os.path.join("data", "sample.txt")
paras = load_paragraphs(data_path)

print(f"Total paragraphs ingested: {len(paras)}\\n")
for i, p in enumerate(paras, 1):
    print(f"=== Paragraph P{i} ===")
    print(p)
    print()
"""))

# Cell 3: Phase 1: Lexical Analysis
cells.append(nbf.v4.new_markdown_cell("""## Step 3: Phase 1 — Lexical Analysis (Lexer & Token Stream Engine)
Analogous to a compiler lexer generating token streams from source code.  
- Sentence segmentation & tokenization
- POS Tagging (`NOUN`, `PROPN`, `VERB`, `ADJ`)
- Base lemmatization & stopword removal without breaking sentence structure
"""))

cells.append(nbf.v4.new_code_cell("""# Test lexical pass on Paragraph 1
doc_p1, tokens_p1, content_p1 = lexical_pass(paras[0])

# Display token stream sample
df_tokens = pd.DataFrame(tokens_p1, columns=["Token Text", "POS Tag", "Lemma"])
print("Lexical Token Stream (First 10 tokens of P1):")
display(df_tokens.head(10))

print("\\nFiltered Content Lemmas (non-stop nouns/verbs/adjectives):")
print(content_p1)
"""))

# Cell 4: Phase 2: Syntactic Analysis
cells.append(nbf.v4.new_markdown_cell("""## Step 4: Phase 2 — Syntactic Analysis & Dependency AST Parsing
Rather than crude word sequence matching, the parser computes sentence dependency syntax trees (ASTs).  
The engine traverses tree branches to identify:
- **Subject**: Nominal subject (`nsubj`, `nsubjpass`), expanded with compound and adjectival modifiers (`span_text`), and handling `conj` coordinates.
- **Predicate**: Verb lemma, capturing negation (`neg`) prefix `"not "`.
- **Object**: Direct object (`dobj`), attribute (`attr`), dative, or prepositional object (`pobj`).
- **Passive Voice**: Correctly swaps semantic patient and agent (e.g. *"The diagnosis was confirmed by the radiologist"* $\\to$ `('radiologist', 'confirm', 'diagnosis')`).
"""))

cells.append(nbf.v4.new_code_cell("""# Demonstration of passive voice syntactic transformation
passive_sample = "The diagnosis was confirmed by the radiologist."
passive_doc = nlp(passive_sample)
print("Passive sentence:", passive_sample)
print("Extracted SPO Triplet:", extract_triplets(passive_doc))

# Render visual AST dependency tree using displaCy
dis状況 = displacy.render(passive_doc, style='dep', jupyter=True, options={'distance': 110})
"""))

# Cell 5: Extract Triplets for all Paragraphs
cells.append(nbf.v4.new_markdown_cell("""### Batch Dependency Parsing Across All Document Paragraphs
Extracting structured relational triplets for paragraphs $P_1$ through $P_4$:"""))

cells.append(nbf.v4.new_code_cell("""raw_trips = [extract_triplets(lexical_pass(p)[0]) for p in paras]

for i, t in enumerate(raw_trips, 1):
    print(f"P{i} Syntactic AST Triplets ({len(t)} found):")
    for trip in t:
        print(f"   -> {trip}")
    print()
"""))

# Cell 6: Phase 3: Semantic Analysis & Symbol Table
cells.append(nbf.v4.new_markdown_cell("""## Step 5: Phase 3 — Semantic Analysis & Symbol Table Coreference
In compiler design, a **Symbol Table** maintains identifier scopes and bindings across blocks.  
Here, the `SymbolTable` maintains cross-paragraph entity tracking:
- Maps anaphoric pronouns (`it`, `they`, `he`, `she`, `these`, `this`) back to their parent nominal entities across paragraph boundaries.
- E.g., in $P_2$: *"It trains the models..."* is resolved to `('company', 'train', 'model')`.
"""))

cells.append(nbf.v4.new_code_cell("""# Execute full multi-pass analysis pipeline
docs, trips, st, M, drift = analyse(paras)

print("=== SYMBOL TABLE REGISTRY ===")
for entity, paragraph_indices in st.entities.items():
    p_labels = [f"P{p+1}" for p in sorted(paragraph_indices)]
    print(f"  • {entity.ljust(25)} -> Scope: {', '.join(p_labels)}")

print("\\n=== COREFERENCE-RESOLVED TRIPLETS ===")
for i, t in enumerate(trips, 1):
    print(f"P{i}: {t}")
"""))

# Cell 7: Coherence Matrix & Semantic Drift
cells.append(nbf.v4.new_markdown_cell("""## Step 6: Multi-Layer Semantic Consistency & Drift Logging
For every paragraph pair $(P_i, P_j)$, the engine evaluates a dual-layer consistency metric:
$$\\text{Score} = 0.4 \\times \\text{Triplet\\_Sim} + 0.6 \\times \\text{Vector\\_Sim}$$
- **Triplet Graph Alignment**: Soft Jaccard similarity across SPO triplets using dense vectors (threshold $\\ge 0.6$).
- **Contextual Vector Similarity**: Cosine similarity of content-only word vectors (preventing stopword dilution).
- **Semantic Drift Detection**: Consecutive transitions with $\\text{Score} < 0.35$ are flagged as logical context breaks.
"""))

cells.append(nbf.v4.new_code_cell("""# Display Consistency Matrix
labels = [f"P{i+1}" for i in range(len(M))]
df_matrix = pd.DataFrame(M, index=labels, columns=labels)
print("=== N x N PARAGRAPH CONSISTENCY MATRIX ===")
display(df_matrix.round(3))

print("\\n=== SEMANTIC DRIFT DIAGNOSTIC LOG (Threshold < 0.35) ===")
if drift:
    for a, b, score in drift:
        print(f"  [!] DRIFT WARNING: Context break detected between P{a} and P{b} (Score: {score:.3f} < {DRIFT_THRESHOLD})")
else:
    print("  [✓] All consecutive transitions meet coherence threshold.")
"""))

# Cell 8: Output Visualizations
cells.append(nbf.v4.new_markdown_cell("""## Step 7: Visualization Layer (Knowledge Graph & Heatmap)
Visualizing document semantics:
1. **Entity-Relationship Knowledge Graph**: Concept entities as nodes, predicate verbs as directed edges.
2. **Consistency Heatmap**: Color-coded matrix showing high consistency in green and semantic drift in red.
"""))

cells.append(nbf.v4.new_code_cell("""import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# 1. Plot Knowledge Graph
plot_knowledge_graph(trips, ax=ax1)

# 2. Plot Consistency Heatmap
plot_heatmap(M, ax=ax2)

plt.tight_layout()
plt.show()

# Save standalone artifacts in outputs directory
os.makedirs("outputs", exist_ok=True)
plot_knowledge_graph(trips, path=os.path.join("outputs", "knowledge_graph.png"))
plot_heatmap(M, path=os.path.join("outputs", "heatmap.png"))
print("High-resolution artifacts saved to outputs/ directory.")
"""))

# Cell 9: Step 8 Add-on 1: Baseline Comparison
cells.append(nbf.v4.new_markdown_cell("""## Step 8 (Add-on 1): Baseline Comparison against Base Paper
**Research Question:** Why is a Compiler AST approach superior to naive word-set Jaccard overlap?  
- **Base Paper:** Uses crude Jaccard overlap on token sets ($|A \\cap B| / |A \\cup B|$). It completely fails on synonymy, active/passive voice, and structural meaning.
- **Our System:** Combines dependency AST triplets, Symbol Table pronoun tracking, and semantic vector embeddings.
"""))

cells.append(nbf.v4.new_code_cell("""# Compare Base Paper (Jaccard) vs Compiler-Guided Engine
df_comparison = generate_baseline_comparison_table(paras, M)
display(df_comparison)
"""))

# Cell 10: Step 8 Add-on 2: Contradiction Detection
cells.append(nbf.v4.new_markdown_cell("""## Step 8 (Add-on 2): Logical Contradiction Detection
Because our parser preserves negation markers (`not`) during AST traversal, the engine can detect contradictory assertions across paragraphs (e.g. $(S, P, O)$ vs $(S, \\text{not } P, O)$).
"""))

cells.append(nbf.v4.new_code_cell("""# Test document with an explicit factual contradiction
contradiction_doc = [
    "The experimental medicine cures acute neurological disorders.",
    "Subsequent clinical research showed that the experimental medicine does not cure acute neurological disorders."
]

c_docs = [lexical_pass(p)[0] for p in contradiction_doc]
_, c_trips = build_symbol_table(c_docs)
contradictions = detect_contradictions(c_trips)

print("Extracted Triplets:")
for i, trips_list in enumerate(c_trips, 1):
    print(f"  Paragraph {i}: {trips_list}")

print("\\nDetected Logical Contradictions:")
for c in contradictions:
    print(f"  [!] {c['Finding']}")
    print(f"      Paragraph {c['Paragraph_A']} Triplet: {c['Triplet_A']}")
    print(f"      Paragraph {c['Paragraph_B']} Triplet: {c['Triplet_B']}")
"""))

# Cell 11: Step 8 Add-on 3: Entity Scope Retention
cells.append(nbf.v4.new_markdown_cell("""## Step 8 (Add-on 3): Entity-Level Scope Retention Tracking
Using the Symbol Table, we track newly introduced concepts versus retained entities across sequential paragraphs. A sudden drop in retention ratio highlights an abrupt topic shift.
"""))

cells.append(nbf.v4.new_code_cell("""df_entity_drift = compute_entity_drift(trips, st)
display(df_entity_drift)
"""))

# Cell 12: Viva Talking Points & Limitations Checklist
cells.append(nbf.v4.new_markdown_cell("""## Viva Talking Points & Defense Checklist

### 1. Connecting Parser ASTs to Natural Language Processing
- In compiler theory, an Abstract Syntax Tree (AST) abstract away concrete syntax (parentheses, spacing) to reveal the grammatical relationship between operators and operands.
- Similarly, our dependency parse trees abstract away word order: an active sentence (*"The radiologist confirmed the diagnosis"*) and passive sentence (*"The diagnosis was confirmed by the radiologist"*) produce the **identical structured triplet** `(radiologist, confirm, diagnosis)`.

### 2. The Symbol Table Analogy
- In programming languages, compilers maintain symbol tables to track variable declarations, scopes, and references.
- In multi-paragraph text analysis, pronouns (*"it"*, *"they"*) act like variable references. Without a Symbol Table, *"it"* in $P_2$ would be treated as an isolated token. Our Symbol Table resolves *"it"* back to its canonical identifier `company`, ensuring semantic continuity across paragraph blocks.

### 3. Dual-Metric Coherence Formulation
- Why not use embeddings alone? Dense vector embeddings capture general topic similarity well, but fail at structural "who-did-what" logic.
- Why not use triplets alone? Exact SPO matching is sparse and misses vocabulary variations.
- Our hybrid formula $\\text{Score} = 0.4 \\times \\text{Triplet\\_Sim} + 0.6 \\times \\text{Vector\\_Sim}$ achieves both semantic breadth and structural precision.

### 4. Known Real-World Limitations
- **Parser ambiguity:** Certain complex sentence structures (e.g., *"AI systems support decision making"*) can lead spaCy to tag *"support"* as a noun modifier rather than root verb, yielding no triplet.
- **Rule-based pronoun resolution:** The current Symbol Table resolves pronouns to the most recent principal subject. Future iterations can integrate neural coreference models (e.g., `fastcoref`).
"""))

nb.cells = cells

# Save notebook file
notebook_path = "c:/Users/Lenovo/Desktop/compiler/coherence_engine/demo.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook successfully written to {notebook_path}")
