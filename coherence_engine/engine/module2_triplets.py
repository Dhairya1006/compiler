"""
Module 2: Syntactic Analysis and Relationship Extraction
Compiler Analogy: Parser (AST Construction) and Intermediate Representation (IR)

Responsibilities:
1. Traverse the spaCy dependency tree as an Abstract Syntax Tree (AST).
2. Extract (Subject, Predicate, Object) triplets representing semantic relations.
3. Contributions implemented:
   - Passive-voice agent swap (Patient was actioned by Agent -> (Agent, action, Patient)).
   - Passive without agent handling (Patient was actioned -> ([unspecified], action, Patient)).
   - "A and B" conjunction expansion for subjects and objects.
   - Coordinated verbs inheriting the subject from their head verb.
   - Negation-aware predicates ("does not reduce" -> "not reduce").
   - Compound and amod modifier expansion ("blood glucose", "hepatic glucose production").
   - Prepositional object extraction for intransitive/relational verbs.
"""

import spacy
from typing import List, Tuple, Union


def span_text(tok: Union[spacy.tokens.Token, str]) -> str:
    """
    Expand a noun token with compound and amod modifiers into a full entity span.
    Preserves modifier order and maintains 'data' without stem mutation.
    Example: 'glucose' with compound 'blood' -> 'blood glucose'
    """
    if isinstance(tok, str):
        return tok

    def get_parts(t):
        res = []
        for c in t.lefts:
            if c.dep_ in {"compound", "amod"}:
                res.extend(get_parts(c))
        res.append(t)
        for c in t.rights:
            if c.dep_ in {"compound"}:
                res.extend(get_parts(c))
        return res

    parts = get_parts(tok)
    words = ["data" if t.lower_ == "data" else t.lemma_.lower() for t in parts]
    return " ".join(words)


def get_subjects(verb: spacy.tokens.Token) -> list:
    """
    Extract nominal subjects (nsubj, nsubjpass) for a given verb.
    Recursively inherits subjects for coordinated verbs ('prescribe ... and monitor ...').
    Expands conjunction subjects ('A and B' via s.conjuncts).
    """
    subs = [c for c in verb.children if c.dep_ in {"nsubj", "nsubjpass"}]
    if not subs and verb.dep_ == "conj":
        return get_subjects(verb.head)

    out = []
    for s in subs:
        out.append(s)
        out.extend(list(s.conjuncts))
    return out


def get_objects(verb: spacy.tokens.Token) -> list:
    """
    Extract objects and complements for a verb:
    - Direct objects (dobj)
    - Attribute complements (attr)
    - Dative objects (dative)
    - Object predicative complements (oprd)
    - Prepositional objects (prep -> pobj), excluding passive agent 'by'
    - Clausal complement objects (xcomp with dobj)
    """
    objs = []
    for c in verb.children:
        # Direct nominal objects and predicative attributes
        if c.dep_ in {"dobj", "attr", "dative", "oprd"}:
            objs.append(c)
            objs.extend(list(c.conjuncts))

        # Prepositional objects (excluding passive 'by')
        elif c.dep_ == "prep" and c.lower_ != "by":
            pobjs = [p for p in c.children if p.dep_ == "pobj"]
            for p in pobjs:
                objs.append(p)
                objs.extend(list(p.conjuncts))

        # Clausal complement handling (e.g. gerund complements like 'increase bleeding risk')
        elif c.dep_ == "xcomp":
            if c.pos_ == "VERB":
                for gc in c.children:
                    if gc.dep_ in {"dobj", "attr", "oprd"}:
                        objs.append(gc)
                        objs.extend(list(gc.conjuncts))

    return objs


def extract_triplets(doc: spacy.tokens.Doc) -> List[Tuple[str, str, str]]:
    """
    Traverse spaCy dependency AST and extract Subject-Predicate-Object triplets.

    Handles:
    - Active voice: (Subject, Predicate, Object)
    - Passive with agent: swaps agent to Subject and syntactic subject to Object
    - Passive without agent: assigns '[unspecified]' to Subject
    - Negation: prepends 'not ' to predicate lemma
    - Conjunction subjects and objects: generates cartesian product of relations
    """
    triplets = []

    for sent in doc.sents:
        for tok in sent:
            # Focus on verbal heads (verbs and predicative auxiliaries)
            if tok.pos_ not in {"VERB", "AUX"}:
                continue
            if tok.pos_ == "AUX" and tok.dep_ != "ROOT":
                continue

            subs = get_subjects(tok)
            objs = get_objects(tok)

            # Check voice: passive if nsubjpass is a child or if verb itself is nsubjpass
            passive = any(c.dep_ == "nsubjpass" for c in tok.children) or tok.dep_ == "nsubjpass"

            # Check negation on verb or inherited from head in coordinated verbs
            neg = any(c.dep_ == "neg" for c in tok.children) or (
                tok.dep_ == "conj" and any(c.dep_ == "neg" for c in tok.head.children)
            )
            pred = ("not " if neg else "") + tok.lemma_.lower()

            if passive:
                # Identify passive agent: 'by <agent>'
                agents = []
                for c in tok.children:
                    if c.dep_ == "agent" or (c.dep_ == "prep" and c.lower_ == "by"):
                        pobjs = [p for p in c.children if p.dep_ == "pobj"]
                        for p in pobjs:
                            agents.append(p)
                            agents.extend(list(p.conjuncts))

                if agents:
                    # Passive voice swap: Agent becomes Subject, syntactic patient becomes Object
                    for a in agents:
                        for s in subs:
                            triplets.append((span_text(a), pred, span_text(s)))
                else:
                    # Passive voice without agent: Agent is unspecified, patient is Object
                    for s in subs:
                        triplets.append(("[unspecified]", pred, span_text(s)))
            else:
                for s in subs:
                    for o in objs:
                        triplets.append((span_text(s), pred, span_text(o)))

    # Deduplicate while preserving order
    return list(dict.fromkeys(triplets))
