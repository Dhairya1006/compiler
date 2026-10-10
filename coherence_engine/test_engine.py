"""
Comprehensive Unit & Integration Test Suite for Coherence Engine
Verifies each compiler stage against ground truth specifications.
"""

import unittest
import numpy as np
import os
import sys

# Ensure local engine can be imported
sys.path.insert(0, os.path.dirname(__file__))
from engine import (
    clean,
    lexical_pass,
    extract_triplets,
    span_text,
    SymbolTable,
    build_symbol_table,
    triplet_similarity,
    vector_similarity,
    analyse,
    detect_contradictions,
    generate_baseline_comparison_table,
    load_paragraphs,
    nlp,
    DRIFT_THRESHOLD
)


class TestCoherenceEngine(unittest.TestCase):

    def test_lexer_pass(self):
        sample = "Solar panels convert sunlight into electricity."
        doc, tokens, content = lexical_pass(sample)
        expected_content = ["solar", "panel", "convert", "sunlight", "electricity"]
        self.assertEqual(content, expected_content)
        self.assertTrue(len(tokens) > 0)

    def test_passive_voice_transformation(self):
        # Passive voice: agent (radiologist) should become subject, patient (diagnosis) should become object
        sample = "The diagnosis was confirmed by the radiologist."
        doc = nlp(sample)
        triplets = extract_triplets(doc)
        self.assertIn(("radiologist", "confirm", "diagnosis"), triplets)

    def test_coordinated_verbs(self):
        # Subject should be inherited across coordinated verbs
        sample = "The company trains the models and deploys them in hospitals."
        doc = nlp(sample)
        triplets = extract_triplets(doc)
        verbs = [p for _, p, _ in triplets]
        self.assertIn("train", verbs)
        self.assertIn("deploy", verbs)

    def test_symbol_table_coreference(self):
        # Pronoun 'it' in P2 should resolve to 'company' from preceding context
        p1 = "The company develops machine learning models."
        p2 = "It trains the models on large datasets."
        docs = [nlp(p1), nlp(p2)]
        st, resolved = build_symbol_table(docs)
        self.assertEqual(st.resolve("it"), "company")
        # In resolved triplets for P2, subject must be 'company'
        p2_subjects = [s for s, _, _ in resolved[1]]
        self.assertIn("company", p2_subjects)

    def test_full_pipeline_and_drift(self):
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample.txt")
        paras = load_paragraphs(sample_path)
        self.assertEqual(len(paras), 4)

        docs, trips, st, M, drift = analyse(paras)

        # Check matrix shape and symmetry
        self.assertEqual(M.shape, (4, 4))
        self.assertTrue(np.allclose(M, M.T))
        self.assertTrue(np.allclose(np.diag(M), 1.0))

        # Check drift detection between P3 and P4
        drift_pairs = [(a, b) for a, b, _ in drift]
        self.assertIn((3, 4), drift_pairs)
        self.assertTrue(M[2, 3] < DRIFT_THRESHOLD)

    def test_contradiction_detection(self):
        contradictory_trips = [
            [("experimental drug", "cause", "severe side effect")],
            [("experimental drug", "not cause", "severe side effect")]
        ]
        contras = detect_contradictions(contradictory_trips)
        self.assertEqual(len(contras), 1)
        self.assertEqual(contras[0]["Paragraph_A"], 1)
        self.assertEqual(contras[0]["Paragraph_B"], 2)


if __name__ == "__main__":
    unittest.main()
