"""
Module 3: Symbol Table, Coreference Resolution, and Entity Drift Tracking
Compiler Analogy: Symbol Table and Scope Analysis (Variable Binding & Scope Tracking)

Responsibilities:
1. Symbol Table / Entity Registry:
   Tracks entities across paragraph scopes, recording occurrence paragraphs,
   first appearance, last appearance, and occurrence frequencies.
2. Rule-Based Pronoun Coreference Resolution:
   Resolves third-person and demonstrative pronouns (it, they, he, she, this, these, those, them)
   to the most recent salient named entity subject.
   Generates an audit log recording every resolution event.
3. Entity Drift Analysis:
   Computes paragraph-to-paragraph entity retention and new-entity ratios:
   New Entity Ratio = |Entities(P_i) \\ Entities(P_{i-1})| / |Entities(P_i)|
4. Optional Neural Coreference Comparison:
   Integrates fastcoref (FCoref) when available, falling back gracefully if dependencies or weights are absent.
"""

from typing import List, Tuple, Dict, Any, Set, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRONOUNS: Set[str] = {"it", "they", "he", "she", "this", "these", "those", "them"}


class SymbolTable:
    """
    Compiler-inspired Symbol Table managing entity scopes and cross-paragraph coreference.
    """

    def __init__(self):
        # Entity -> dict with occurrence metadata
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.last_named_subject: Optional[str] = None
        self.resolution_log: List[Dict[str, Any]] = []

    def register_entity(self, entity: str, paragraph_id: int) -> None:
        """Register an entity occurrence within a specific paragraph scope."""
        entity = entity.strip().lower()
        if not entity or entity in PRONOUNS or entity == "[unspecified]":
            return

        if entity not in self.registry:
            self.registry[entity] = {
                "paragraphs": {paragraph_id},
                "first_seen": paragraph_id,
                "last_seen": paragraph_id,
                "frequency": 1
            }
        else:
            self.registry[entity]["paragraphs"].add(paragraph_id)
            self.registry[entity]["last_seen"] = paragraph_id
            self.registry[entity]["frequency"] += 1

    def resolve_pronoun(self, token_text: str, paragraph_id: int, role: str = "subject") -> str:
        """
        Resolve anaphoric pronoun to the active named subject in the symbol table scope.
        Logs each resolution attempt for review and verification.
        """
        token_lower = token_text.strip().lower()
        if token_lower in PRONOUNS and self.last_named_subject:
            resolved = self.last_named_subject
            self.resolution_log.append({
                "paragraph_id": paragraph_id,
                "pronoun": token_text,
                "resolved_entity": resolved,
                "role": role,
                "status": "RESOLVED"
            })
            return resolved

        if token_lower in PRONOUNS and not self.last_named_subject:
            self.resolution_log.append({
                "paragraph_id": paragraph_id,
                "pronoun": token_text,
                "resolved_entity": token_text,
                "role": role,
                "status": "UNRESOLVED (no prior antecedent)"
            })
            return token_text

        # Not a pronoun - update active scope's named subject if this is a subject
        if role == "subject" and token_lower not in PRONOUNS and token_lower != "[unspecified]":
            self.last_named_subject = token_lower

        return token_text

    def get_registry_dataframe(self) -> pd.DataFrame:
        """Return the entity registry formatted as a pandas DataFrame."""
        rows = []
        for entity, data in sorted(self.registry.items(), key=lambda x: x[1]["frequency"], reverse=True):
            rows.append({
                "Entity": entity,
                "Frequency": data["frequency"],
                "First_Seen_P": data["first_seen"],
                "Last_Seen_P": data["last_seen"],
                "Paragraphs_Present": sorted(list(data["paragraphs"]))
            })
        return pd.DataFrame(rows)

    def get_resolution_log_dataframe(self) -> pd.DataFrame:
        """Return the coreference resolution log as a pandas DataFrame."""
        return pd.DataFrame(self.resolution_log)


def resolve_triplets_with_symbol_table(
    triplets_per_paragraph: List[List[Tuple[str, str, str]]]
) -> Tuple[List[List[Tuple[str, str, str]]], SymbolTable]:
    """
    Process extracted triplets sequentially through the Symbol Table to:
    1. Register named entities.
    2. Resolve pronoun subjects to their antecedent.
    3. Return resolved triplets and the populated Symbol Table.
    """
    st = SymbolTable()
    resolved_all = []

    for pid, trips in enumerate(triplets_per_paragraph, start=1):
        resolved_p = []
        for s, p, o in trips:
            # First register entities if named
            st.register_entity(s, pid)
            st.register_entity(o, pid)

            # Resolve subject pronoun
            res_s = st.resolve_pronoun(s, pid, role="subject")
            res_o = st.resolve_pronoun(o, pid, role="object") if o.lower() in PRONOUNS else o

            # If subject was resolved, register the resolved entity too
            if res_s != s:
                st.register_entity(res_s, pid)

            resolved_p.append((res_s, p, res_o))
        resolved_all.append(resolved_p)

    return resolved_all, st


def compute_entity_drift(triplets_per_paragraph: List[List[Tuple[str, str, str]]]) -> pd.DataFrame:
    """
    Compute paragraph-level entity continuity and drift scores.
    For paragraph P_i:
    - New Entity Ratio = |Entities(P_i) \\ Entities(P_{i-1})| / |Entities(P_i)|
    - Retention Ratio = |Entities(P_i) ∩ Entities(P_{i-1})| / |Entities(P_i)|
    """
    records = []
    prev_entities: Set[str] = set()

    for pid, trips in enumerate(triplets_per_paragraph, start=1):
        current_entities = set()
        for s, _, o in trips:
            if s not in PRONOUNS and s != "[unspecified]":
                current_entities.add(s.lower())
            if o not in PRONOUNS and o != "[unspecified]":
                current_entities.add(o.lower())

        if pid == 1:
            new_entities = current_entities
            retained_entities = set()
            new_ratio = 1.0 if current_entities else 0.0
            retention_ratio = 0.0
        else:
            retained_entities = current_entities.intersection(prev_entities)
            new_entities = current_entities - prev_entities
            total = len(current_entities)
            new_ratio = len(new_entities) / total if total > 0 else 0.0
            retention_ratio = len(retained_entities) / total if total > 0 else 0.0

        records.append({
            "Paragraph": f"P{pid}",
            "Paragraph_ID": pid,
            "Total_Entities": len(current_entities),
            "Retained_Entities_Count": len(retained_entities),
            "New_Entities_Count": len(new_entities),
            "Retention_Ratio": round(retention_ratio, 3),
            "New_Entity_Ratio": round(new_ratio, 3),
            "Retained_List": sorted(list(retained_entities)),
            "New_List": sorted(list(new_entities))
        })

        prev_entities = current_entities

    return pd.DataFrame(records)


def plot_entity_drift(drift_df: pd.DataFrame, output_path: str = "outputs/module3_entity_drift.png", ax=None) -> None:
    """
    Generate line plot displaying entity retention ratio and new-entity ratio across paragraphs.
    Saves figure at 300 dpi.
    """
    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        save_fig = True

    x = drift_df["Paragraph"]
    ax.plot(x, drift_df["Retention_Ratio"], marker="o", color="#2b8a3e", linewidth=2.2, label="Retention Ratio (Coherence)")
    ax.plot(x, drift_df["New_Entity_Ratio"], marker="s", color="#e03131", linewidth=2.2, linestyle="--", label="New Entity Ratio (Drift)")

    ax.set_title("Module 3: Entity Continuity & Drift Dynamics Across Paragraphs", fontsize=11, fontweight="bold")
    ax.set_xlabel("Paragraph Scope", fontsize=10)
    ax.set_ylabel("Entity Ratio", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True)

    if save_fig:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def compare_with_fastcoref(text: str) -> Dict[str, Any]:
    """
    Optional comparison: Runs neural fastcoref resolver on the input text.
    Wrapped in try/except to prevent runtime crashes if weights cannot be downloaded.
    """
    try:
        from fastcoref import FCoref
        model = FCoref(device="cpu")
        preds = model.predict(texts=[text])
        clusters = preds[0].get_clusters(as_strings=True)
        return {
            "status": "SUCCESS",
            "model": "fastcoref (FCoref)",
            "clusters_found": len(clusters),
            "clusters": clusters
        }
    except Exception as e:
        return {
            "status": "SKIPPED",
            "model": "fastcoref",
            "reason": f"Neural coreference skipped: {str(e)}"
        }
