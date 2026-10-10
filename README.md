# Compiler-Guided Dependency Tree Triplet Extraction & Contextual Coherence Engine

**Team No.:** 3  
**Team Members:** Raghav Arora (24BCE2299) & Dhairya Sawant (24BCE2319)  
**Subject:** Compiler Design / Natural Language Processing  

---

## 1. Executive Summary & Compiler Phase Mapping

In compiler design, source code is systematically verified through Lexical, Syntactic, and Semantic stages:
1. **Lexical Analysis (Lexer):** Converts character streams into structured tokens and strips whitespace/comments.
2. **Syntactic Analysis (Parser & AST):** Evaluates grammar rules to construct an Abstract Syntax Tree (AST).
3. **Semantic Analysis (Symbol Table & Type Checker):** Tracks variable bindings and scope across blocks, and ensures type compatibility.

This project implements a direct structural equivalent for natural language documents:
- **Lexical Pass:** Sentence segmentation, tokenization, POS tagging (`NOUN`, `VERB`, `ADJ`), lemmatization, and stopword filtering.
- **Syntactic Pass (AST Parser):** Dependency parse tree traversal extracting Subject–Predicate–Object (SPO) relational triplets across active/passive clauses and compound modifiers.
- **Semantic Pass (Symbol Table & Coherence):** Cross-paragraph pronoun resolution (Symbol Table) + hybrid contextual coherence scoring combining dense vector similarity with soft-Jaccard triplet alignment.
- **Diagnostic / Error Layer:** Identifies logical context breaks ($< 0.35$ coherence score) as Semantic Drift warnings.

---

## 2. Project Directory Structure

```text
compiler/
├── venv/                              # Isolated virtual environment
├── coherence_engine/
│   ├── engine.py                      # Core compiler engine implementation (Steps 2-6 + Add-ons)
│   ├── demo.ipynb                     # Fully executed Jupyter Notebook with rendered outputs
│   ├── test_engine.py                 # Automated unit and integration test suite
│   ├── build_notebook.py              # Generator script for demo.ipynb
│   ├── data/
│   │   └── sample.txt                 # Multi-paragraph benchmark document
│   └── outputs/
│       ├── knowledge_graph.png        # NetworkX entity-relationship graph
│       └── heatmap.png                # Seaborn pairwise consistency heatmap
└── README.md                          # Project documentation and Viva guide
```

---

## 3. Quickstart & Execution

### Activate Virtual Environment
```bash
# Windows PowerShell / CMD
.\venv\Scripts\activate
```

### Run Unit Tests
```bash
python coherence_engine/test_engine.py
```

### Launch Jupyter Notebook
```bash
jupyter notebook coherence_engine/demo.ipynb
```

---

## 4. Benchmark Results on Test Document

### Triplet Extraction & Symbol Table Resolution
- **P1:** `('artificial intelligence', 'optimize', 'complex data analysis')`
- **P2:** Pronoun *"it"* resolved to *"company"* via Symbol Table:
  - `('company', 'develop', 'learning model')`
  - `('company', 'train', 'model')`
  - `('company', 'deploy', 'hospital')`
- **P3:** Passive sentence *"The diagnosis was confirmed by the radiologist"* converted to:
  - `('doctor', 'use', 'model')`
  - `('radiologist', 'confirm', 'diagnosis')`
- **P4:** Off-topic paragraph on solar panels:
  - `('solar panel', 'convert', 'sunlight')`
  - `('solar panel', 'convert', 'clean electricity')`
  - `('solar panel', 'convert', 'home')`

### Coherence Matrix ($N \times N$)
| | P1 | P2 | P3 | P4 |
|---|:---:|:---:|:---:|:---:|
| **P1** | 1.000 | 0.490 | 0.388 | 0.388 |
| **P2** | 0.490 | 1.000 | 0.423 | 0.368 |
| **P3** | 0.388 | 0.423 | 1.000 | **0.278** |
| **P4** | 0.388 | 0.368 | **0.278** | 1.000 |

### Semantic Drift Diagnostic Log
`[!] WARNING: semantic drift between P3 and P4 (score 0.278 < 0.35)`  
The solar panel paragraph is accurately flagged as an off-topic divergence.

---

## 5. Viva Talking Points Checklist

1. **Why Dependency Trees as ASTs?**  
   Like compiler ASTs abstracting away syntax noise, dependency trees abstract word order. Active and passive voice produce identical semantic triplets.
2. **Why a Symbol Table?**  
   Pronouns (*"it"*, *"they"*) act like identifier references. Without a Symbol Table, cross-paragraph references break.
3. **Why Hybrid Scoring ($0.4 \times \text{triplet} + 0.6 \times \text{vector}$)?**  
   Embeddings capture general topic but ignore grammar structure. Triplets capture grammar structure but are sparse. The hybrid formula combines both.
4. **Why Better Than the Base Paper?**  
   The base paper uses naive token Jaccard overlap, which drops to $0.00$ on synonyms and passive structures. Our system maintains accurate semantic affinity scores.
