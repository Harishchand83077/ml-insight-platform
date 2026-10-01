"""
Builds the local knowledge base query_project_docs_tool retrieves from:
loads docs/glossary.md, docs/decisions.md, and every .md file under
docs/knowledge_base/ (synthetic Vantrix Communications telecom policy
docs, retention playbooks, the model card, and the customer_features
data dictionary), splits each one by markdown heading
(MarkdownHeaderTextSplitter) with a token-count size cap on top
(RecursiveCharacterTextSplitter, same as before) for any section that's
still too large for one chunk, embeds each chunk with
sentence-transformers (BAAI/bge-small-en-v1.5, runs locally, no API
calls/cost), and persists them into a local Chroma vector store at
data/chroma_db/.

Splitting by heading first (instead of only by raw character/token
count, as the original version did) keeps each chunk's content aligned
with one coherent section - e.g. a chunk never straddles the boundary
between "Early Termination Fee" and the next policy's "Refund Method"
section - and lets every chunk carry its section heading as metadata,
which query_project_docs_tool uses to build a citation label like
"[source: contract_terms.md > 2-Year Contract]" instead of just a bare
filename.

The knowledge_base/ docs each start with YAML front matter (title,
doc_type, version, last_updated); glossary.md and decisions.md predate
that convention and have none, so they fall back to a fixed doc_type
(DEFAULT_DOC_TYPES below). Front matter is parsed with a tiny
hand-rolled parser rather than pulling in PyYAML as a new runtime
dependency - these front-matter blocks are flat `key: value` pairs
only, never nested structures, so a full YAML parser is more dependency
than the format needs.

Run this once after docs/glossary.md, docs/decisions.md, or anything
under docs/knowledge_base/ changes:
    python src/agent/build_knowledge_base.py

If you change EMBEDDING_MODEL, delete data/chroma_db/ first and rebuild
from scratch - embeddings from different models live in different vector
spaces and must not be mixed in the same collection.
"""

import re
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOCS_DIR = Path("docs")
KNOWLEDGE_BASE_DIR = DOCS_DIR / "knowledge_base"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "project_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE_TOKENS = 480
CHUNK_OVERLAP_TOKENS = 50
# BAAI/bge-small-en-v1.5 has a 512-token max sequence length (verified via
# both its tokenizer.model_max_length and SentenceTransformer.max_seq_length -
# they agree). The splitter counts length via tokenizer.tokenize(), which
# does NOT include the [CLS]/[SEP] special tokens the model adds at embed
# time, so a chunk_size=500 chunk would actually encode to ~502 tokens -
# only a 10-token margin under 512. Trimmed to 480 for a comfortable ~30
# token buffer instead.

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# glossary.md and decisions.md predate the front-matter convention used
# under knowledge_base/ - give them a fixed doc_type instead.
DEFAULT_DOC_TYPES = {
    "docs/glossary.md": "glossary",
    "docs/decisions.md": "decision_log",
}

_FRONT_MATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def parse_front_matter(text):
    """Minimal flat `key: value` front-matter parser - these docs only
    ever use scalar fields (title, doc_type, version, last_updated), so
    a full YAML parser isn't needed. Returns (front_matter_dict,
    body_text_with_front_matter_stripped); front_matter is {} and body
    is the whole text unchanged if there's no front-matter block."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text

    front_matter = {}
    for line in match.group(1).splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        front_matter[key.strip()] = value

    return front_matter, text[match.end() :]


def discover_source_files():
    """docs/glossary.md and docs/decisions.md (hand-authored, no front
    matter), plus every .md file under docs/knowledge_base/ (each with
    YAML front matter), sorted for a reproducible chunk order."""
    fixed = [DOCS_DIR / "glossary.md", DOCS_DIR / "decisions.md"]
    knowledge_base_docs = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
    return fixed + knowledge_base_docs


def load_documents():
    """Returns a list of (source_path, doc_type, front_matter, body) for
    every discovered source file."""
    loaded = []
    for path in discover_source_files():
        text = path.read_text(encoding="utf-8")
        front_matter, body = parse_front_matter(text)
        source_path = path.as_posix()
        doc_type = front_matter.get("doc_type", DEFAULT_DOC_TYPES.get(source_path, "unknown"))
        loaded.append((source_path, doc_type, front_matter, body))
    return loaded


def chunk_documents(documents, embeddings):
    """Splits each document's body by markdown heading first, then caps
    any still-too-large section with the token-based character splitter
    - most sections land well under the cap and pass through that second
    split unchanged. Each output chunk carries source file, doc_type, and
    its most specific heading (section) as metadata."""
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embeddings._client.tokenizer,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    chunks = []
    for source_path, doc_type, front_matter, body in documents:
        header_sections = header_splitter.split_text(body)
        fallback_section = front_matter.get("title", source_path)

        for header_section in header_sections:
            section = (
                header_section.metadata.get("h3")
                or header_section.metadata.get("h2")
                or header_section.metadata.get("h1")
                or fallback_section
            )
            for sized_text in size_splitter.split_text(header_section.page_content):
                chunks.append(
                    Document(
                        page_content=sized_text,
                        metadata={"source": source_path, "doc_type": doc_type, "section": section},
                    )
                )

    return chunks


def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    documents = load_documents()
    chunks = chunk_documents(documents, embeddings)
    print(f"Loaded {len(documents)} files -> split into {len(chunks)} chunks")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Persisted {len(chunks)} chunks to {CHROMA_DIR} (collection '{COLLECTION_NAME}')")

    if len(chunks) == 0:
        print("ERROR: 0 chunks produced - refusing to leave an empty index in place.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
