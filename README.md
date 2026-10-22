# Compiler-Guided Dependency Tree Triplet Extraction and Coherence Engine

Compiler Design / Natural Language Processing project by **Team 3**:
**Raghav Arora (24BCE2299)** and **Dhairya Sawant (24BCE2319)**.

The project applies the stages of a compiler to multi-paragraph natural-language
documents. It extracts canonical Subject-Predicate-Object (SPO) relations,
tracks entities across paragraph scopes, measures contextual coherence, and
reports semantic drift and logical contradictions.

## How the compiler analogy maps to the implementation

| Compiler stage | NLP implementation | Main module |
| --- | --- | --- |
| Lexical analysis | Paragraph ingestion, tokenization, POS tagging, lemmatization, stopword filtering, and content-word extraction | [`module1_lexical.py`](coherence_engine/engine/module1_lexical.py) |
| Syntax analysis / AST construction | spaCy dependency-tree traversal and canonical SPO triplet extraction | [`module2_triplets.py`](coherence_engine/engine/module2_triplets.py) |
| Semantic analysis / symbol table | Entity registration, paragraph-scope metadata, heuristic pronoun resolution, and entity drift metrics | [`module3_symbol_table.py`](coherence_engine/engine/module3_symbol_table.py) |
| Verification and diagnostics | Hybrid coherence scoring, threshold calibration, contradiction detection, drift logs, and visualizations | [`module4_scoring.py`](coherence_engine/engine/module4_scoring.py) |

## Features

- Reads `.txt` and `.docx` documents and keeps paragraphs containing at least
  five words.
- Extracts normalized triplets from active and passive clauses.
- Handles compound and adjectival noun modifiers, coordinated subjects and
  objects, coordinated verbs, prepositional objects, agentless passives, and
  syntactic negation.
- Normalizes passive voice, for example:

  ```text
  Blood glucose is reduced by metformin.
  -> (metformin, reduce, blood glucose)
  ```

- Resolves `it`, `they`, `he`, `she`, `this`, `these`, `those`, and `them` to
  the most recent salient named subject using the symbol table.
- Computes a hybrid score:

  ```text
  coherence = 0.4 * soft_triplet_similarity + 0.6 * embedding_similarity
  ```

- Supports spaCy `en_core_web_md` vectors and optional
  `all-MiniLM-L6-v2` Sentence-Transformers embeddings. If the latter is not
  available, scoring falls back to spaCy vectors.
- Detects consecutive paragraph transitions below the default drift threshold
  of `0.35`.
- Detects contradictions when subject and object match and one predicate is
  negated while the other is not.
- Produces coherence heatmaps, entity-drift plots, knowledge graphs, CSV
  reports, and diagnostic logs.

## Repository layout

```text
compiler/
├── README.md
├── results_summary.md
├── colab_full_project.ipynb
├── coherence_engine/
│   ├── engine/                 # Maintained modular implementation
│   │   ├── __init__.py         # Unified public API
│   │   ├── module1_lexical.py
│   │   ├── module2_triplets.py
│   │   ├── module3_symbol_table.py
│   │   └── module4_scoring.py
│   ├── data/                   # Corpora and evaluation data
│   ├── outputs/                # Generated CSV, PNG, and log artifacts
│   ├── tests/
│   │   └── test_module2.py
│   ├── experiments.py          # Benchmark and end-to-end runner
│   ├── build_notebook.py       # Generates demo.ipynb
│   ├── demo.ipynb
│   └── notebooks/
├── data/                       # Additional top-level copies of benchmark data
└── outputs/                    # Additional top-level output artifacts
```

The modular package in `coherence_engine/engine/` is the implementation used
by the benchmark runner. The older top-level `coherence_engine/engine.py` and
`coherence_engine/test_engine.py` are retained repository files, but the
modular API and `coherence_engine/tests/test_module2.py` are the current
workflow.

## Setup

The repository includes a `venv` directory in the working copy. Activate it
from the repository root, or use another Python 3.10+ environment:

```powershell
.\venv\Scripts\Activate.ps1
```

The runtime requires Python packages including `spacy`, `numpy`, `pandas`,
`networkx`, `matplotlib`, `seaborn`, `scikit-learn`, `sentence-transformers`,
`python-docx`, `jupyter`, and `nbformat`. The spaCy model used by the project
must also be installed:

```powershell
python -m spacy download en_core_web_md
```

`sentence-transformers` is optional at runtime; the scoring module explicitly
falls back to spaCy vectors when `all-MiniLM-L6-v2` cannot be loaded.

## Run the tests

From the repository root:

```powershell
python coherence_engine\tests\test_module2.py
```

The test cases cover simple active clauses, active/passive equivalence,
agentless passives, conjunction expansion, coordinated verbs, negation,
prepositional objects, multi-clause input, compound spans, and gerund parses.

## Run the benchmarks and generate outputs

Run all benchmark evaluations and end-to-end examples:

```powershell
python coherence_engine\experiments.py
```

The script performs the following:

1. Evaluates triplet extraction against 50 biomedical sentences in
   `coherence_engine/data/gold_triplets.json`.
2. Compares seven scoring variants on 40 labelled paragraph pairs in
   `coherence_engine/data/labelled_pairs.csv`.
3. Calibrates drift thresholds from `0.25` through `0.50`.
4. Runs end-to-end examples for metformin, warfarin/aspirin, and ibuprofen.
5. Runs the injected metformin contradiction example.
6. Exports symbol-table, pronoun-resolution, entity-drift, calibration, and
   visualization artifacts to `coherence_engine/outputs/`.

To generate the demonstration notebook:

```powershell
python coherence_engine\build_notebook.py
jupyter notebook coherence_engine\demo.ipynb
```

## Datasets

The benchmark inputs are in [`coherence_engine/data`](coherence_engine/data):

| File | Purpose |
| --- | --- |
| `gold_triplets.json` | 50 biomedical sentences with 58 annotated triplets for precision, recall, and F1 evaluation |
| `labelled_pairs.csv` | 40 labelled paragraph pairs: 15 consistent, 15 drifting, and 10 contradicting |
| `metformin_glucose.txt` | Four-paragraph active/passive and mechanism-consistency example |
| `metformin_glucose_contradiction.txt` | The same topic with a fifth paragraph containing explicit negated claims |
| `warfarin_aspirin_bleeding.txt` | Consistent antithrombotic and bleeding-risk example |
| `warfarin_aspirin_drift.txt` | Warfarin/aspirin example containing an off-topic drift |
| `ibuprofen_drift.txt` | Ibuprofen discussion with an injected solar-panel paragraph |
| `sample.txt` | Small four-paragraph demonstration document |

## Current benchmark results

The checked-in generated summaries report:

- **Triplet extraction:** precision `0.927`, recall `0.879`, F1 `0.903`
  on the 50-sentence gold benchmark.
- **Best calibrated drift threshold:** `0.45`, with accuracy, precision,
  recall, and F1 all reported as `1.000` on the labelled-pair benchmark.
- **Sentence-Transformers baseline:** F1 `0.968` for drift detection.
- **Proposed hybrid (SBERT + AST):** F1 `0.789` for drift detection.
- **AST contradiction detection:** precision `1.000`, recall `0.600`, and
  F1 `0.750` on the injected contradiction evaluation.

Detailed methodology, calibration tables, comparative results, and error
analysis are available in [`coherence_engine/results_summary.md`](coherence_engine/results_summary.md).

## Public Python API

The package re-exports its supported functions from
[`coherence_engine/engine/__init__.py`](coherence_engine/engine/__init__.py):

```python
from coherence_engine.engine import (
    load_paragraphs,
    lexical_analysis,
    extract_triplets,
    resolve_triplets_with_symbol_table,
    compute_hybrid_score,
    build_coherence_matrix,
    detect_semantic_drift,
    detect_contradictions,
    flag_document_anomalies,
)
```

For example, a document can be analyzed with:

```python
from coherence_engine.engine import (
    build_coherence_matrix,
    detect_semantic_drift,
    load_paragraphs,
)

paragraphs = load_paragraphs("coherence_engine/data/ibuprofen_drift.txt")
matrix, docs, triplets = build_coherence_matrix(
    paragraphs,
    use_sbert=True,
)
drift_events = detect_semantic_drift(matrix)
```

`build_coherence_matrix` returns the symmetric paragraph-by-paragraph score
matrix, processed spaCy documents, and extracted triplets. The unified
`flag_document_anomalies` helper additionally returns drift events,
contradictions, and aggregate anomaly counts.
