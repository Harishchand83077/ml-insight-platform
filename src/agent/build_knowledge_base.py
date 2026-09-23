"""
Builds the local knowledge base the query_project_docs_tool retrieves
from: loads docs/glossary.md and docs/decisions.md, splits them into
~480-token chunks (with overlap, measured using the actual embedding
model's own tokenizer), embeds each chunk with sentence-transformers
(BAAI/bge-small-en-v1.5, runs locally, no API calls/cost), and persists
them into a local Chroma vector store at data/chroma_db/.

Run this once after docs/glossary.md or docs/decisions.md change:
python src/agent/build_knowledge_base.py

If you change EMBEDDING_MODEL, delete data/chroma_db/ first and rebuild
from scratch - embeddings from different models live in different vector
spaces and must not be mixed in the same collection.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOURCE_FILES = ["docs/glossary.md", "docs/decisions.md"]
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


def load_documents():
    docs = []
    for path in SOURCE_FILES:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": path}))
    return docs


def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embeddings._client.tokenizer,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    documents = load_documents()
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} files -> split into {len(chunks)} chunks")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Persisted {len(chunks)} chunks to {CHROMA_DIR} (collection '{COLLECTION_NAME}')")


if __name__ == "__main__":
    main()
