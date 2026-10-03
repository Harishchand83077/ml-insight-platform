"""
NOT CURRENTLY WIRED INTO query_project_docs_tool - src/agent/tools.py
uses plain vectorstore.similarity_search() instead. This module is kept
implemented, tested (tests/eval/rag_eval.py --hybrid), and documented
because measuring it is what justified not using it: see
docs/decisions.md's "Hybrid search evaluated, not adopted" entry and
tools.py's own module docstring for the numbers. Re-enabling it is one
line in tools.py's query_project_docs_tool if a future corpus or
question mix changes that calculus.

Hybrid (BM25 + vector) retrieval for query_project_docs_tool, fused with
weighted Reciprocal Rank Fusion (RRF) rather than LangChain's
EnsembleRetriever/BM25Retriever - both live in langchain_community, a
much larger, broadly-scoped package this project has deliberately never
pulled in (requirements-docker.txt stays a curated ~50-package subset of
the full dev environment, see its own docstring). rank_bm25's BM25Okapi
is the one new, genuinely small dependency this adds (pure Python,
depends only on numpy, already required).

Why RRF over a manual weighted-score combination: BM25 scores (Okapi
term-frequency weighted, unbounded - can be 0 to 20+ depending on corpus
and query) and Chroma's cosine relevance scores (normalized to [0, 1])
live on completely different scales. Averaging raw scores directly would
let whichever retriever happens to produce larger numbers dominate the
result, not whichever one actually found the better match - a score of
"8.3" from BM25 isn't three times more confident than a "0.3" from
vector search, those numbers aren't comparable at all. RRF sidesteps
this by discarding raw scores entirely and fusing on RANK POSITION only:

    fused_score(chunk) = sum over retrievers of weight / (RRF_K + rank)

A chunk ranked #1 by BM25 and #1 by vector search contributes the same
amount regardless of what either retriever's raw score happened to be.
RRF_K=60 is the standard constant from the original RRF paper (Cormack,
Clarke & Buettcher, 2009) and LangChain's own EnsembleRetriever default -
large enough that a #1 vs #2 ranking difference isn't wildly
overweighted, small enough that top ranks still dominate low ones.

BM25_WEIGHT / VECTOR_WEIGHT are exposed as top-level constants
specifically so they're easy to find and re-tune, then re-measure with
tests/eval/rag_eval.py, without touching the fusion logic itself.

The BM25 index is built once per process (lazily, on first call) over
the exact same chunks build_knowledge_base.py already persisted into
Chroma - by calling that module's own load_documents()/chunk_documents()
a second time, not by re-deriving a second chunking from docs/ some
other way. Both indexes are therefore guaranteed to agree on what a
"chunk" is; they can disagree on how to rank one, which is the whole
point of combining them. Each chunk's chunk_id metadata field (set in
build_knowledge_base.py) is what lets a chunk BM25 and Chroma each
surface independently get recognized as "the same chunk" during fusion.
"""

from rank_bm25 import BM25Okapi

from src.agent.build_knowledge_base import chunk_documents, load_documents

RRF_K = 60
# 0.5/0.5 measured (tests/eval/rag_eval.py, reports/rag_eval_hybrid_50_50.json)
# worse than vector-only on every category: exact-term hit@1 unchanged
# (0.67), paraphrase hit@1 dropped 0.875->0.75, and 2/3 unanswerable
# questions scored above the suspicious threshold vs. 1/3 for vector-only
# - BM25's keyword overlap was inflating confidence on topically-related
# but actually-irrelevant chunks. Shifted toward BM25 once (0.7/0.3,
# exact-term matching is BM25's actual strength) per the one-time-retune
# instruction this was built under - see reports/rag_eval_hybrid.json for
# the result that weighting was kept or reverted against.
BM25_WEIGHT = 0.7
VECTOR_WEIGHT = 0.3
# How many top candidates each individual retriever contributes to the
# fusion pool, wider than the final k so a chunk that's merely decent on
# one side but strong on the other still has a chance to be pulled in by
# the other retriever's ranking.
CANDIDATE_POOL_SIZE = 10

_bm25_index = None
_bm25_chunks = None


def _tokenize(text):
    return text.lower().split()


def _ensure_bm25_index(embeddings):
    """Builds the module-level BM25 index on first use - mirrors how
    src/agent/tools.py lazily builds the Chroma vectorstore via a
    module-level _vectorstore. embeddings is only used for
    chunk_documents()'s text-splitting tokenizer, not for computing any
    embedding vectors - BM25 itself is a pure term-frequency index and
    doesn't embed anything."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return
    _bm25_chunks = chunk_documents(load_documents(), embeddings)
    _bm25_index = BM25Okapi([_tokenize(chunk.page_content) for chunk in _bm25_chunks])


def _fuse(vectorstore, embeddings, query, k):
    """Returns the top-k (chunk, fused_score) pairs. fused_score is the
    raw weighted-RRF sum, not normalized - see normalized_top_score below
    for a [0, 1]-ish version comparable across queries."""
    _ensure_bm25_index(embeddings)

    vector_hits = vectorstore.similarity_search(query, k=CANDIDATE_POOL_SIZE)

    bm25_scores = _bm25_index.get_scores(_tokenize(query))
    bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_hits = [_bm25_chunks[i] for i in bm25_ranked_idx[:CANDIDATE_POOL_SIZE]]

    fused_scores = {}
    chunk_by_id = {}

    for rank, doc in enumerate(vector_hits):
        chunk_id = doc.metadata["chunk_id"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + VECTOR_WEIGHT / (RRF_K + rank)
        chunk_by_id[chunk_id] = doc

    for rank, doc in enumerate(bm25_hits):
        chunk_id = doc.metadata["chunk_id"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + BM25_WEIGHT / (RRF_K + rank)
        chunk_by_id.setdefault(chunk_id, doc)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
    return [(chunk_by_id[cid], fused_scores[cid]) for cid in ranked_ids]


def max_possible_rrf_score():
    """The fused_score a chunk would get if it ranked #1 (rank=0) with
    BOTH retrievers simultaneously - used to normalize a raw fused_score
    into a [0, 1]-ish "fraction of maximum possible confidence" that's
    meaningful to compare across queries and against a fixed threshold,
    the way Chroma's own relevance scores already are."""
    return (BM25_WEIGHT + VECTOR_WEIGHT) / RRF_K


def hybrid_search(vectorstore, embeddings, query, k=3):
    """Returns the top-k chunks by weighted RRF over BM25 and vector
    search - the function query_project_docs_tool calls."""
    return [doc for doc, _ in _fuse(vectorstore, embeddings, query, k)]


def hybrid_search_with_scores(vectorstore, embeddings, query, k=3):
    """Same as hybrid_search, but also returns each chunk's fused_score
    normalized by max_possible_rrf_score() into [0, 1]-ish - used by
    tests/eval/rag_eval.py's unanswerable-question "suspiciously high
    score" check, so that check means roughly the same thing (a fraction
    of this retriever's own maximum possible confidence) whether it's
    evaluating the vector-only baseline or this hybrid retriever, even
    though the two underlying score types aren't otherwise comparable."""
    max_score = max_possible_rrf_score()
    return [(doc, score / max_score) for doc, score in _fuse(vectorstore, embeddings, query, k)]
