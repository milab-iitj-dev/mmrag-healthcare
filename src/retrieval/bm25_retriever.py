"""
BM25 sparse text retrieval over medical reports.

Uses the BM25 algorithm (Okapi BM25) to retrieve the most relevant
radiology reports given a text query. This is the simplest retrieval
baseline and works well for exact keyword matching.

Implementation: Phase 3
"""

# TODO: Implement BM25 retrieval using rank_bm25 or custom implementation
# - index(): tokenise and build inverted index over report text
# - retrieve(): score query against all documents, return top-k
# - save_index() / load_index(): pickle the internal state
