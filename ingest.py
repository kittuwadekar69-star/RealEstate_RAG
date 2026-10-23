
from pathlib import Path
import re
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
DATA = BASE / "data"
DB_DIR = BASE / "chroma_db"
COLLECTION_NAME = "prompt_engineering"

# Local embedding model; downloads once on first run.
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdfs():
    documents = []

    for path in sorted(DATA.glob("*.pdf")):
        print(f"Reading: {path.name}")
        reader = PdfReader(str(path))

        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                documents.append({
                    "text": text,
                    "source": path.name,
                    "page": page_no
                })

    return documents

def chunk_text(text, chunk_size=1800, overlap=300):
    # Character-based chunks; approximately 300–800 tokens
    # for typical prose, but verify chunk sizes during testing.
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def extract_text_files():
    documents = []
    for path in sorted(DATA.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for page_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if line:
                documents.append({
                    "text": line,
                    "source": path.name,
                    "page": page_no
                })
    return documents


def main():
    if not list(DATA.glob("*.pdf")) and not list(DATA.glob("*.txt")):
        raise SystemExit("No PDF or TXT files found in data.")

    pages = extract_pdfs()+extract_text_files()
    records = []

    for page in pages:
        for index, chunk in enumerate(chunk_text(page["text"])):
            records.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"],
                "chunk": index
            })

    if not records:
        raise SystemExit(
            "No extractable PDF text found. The PDFs may be scanned images."
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Rebuild collection so repeated ingestion doesn't create duplicates.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 64
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        texts = [r["text"] for r in batch]
        embeddings = embedder.encode(
            texts, normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=[
                f"doc_{start + i}"
                for i in range(len(batch))
            ],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": r["source"],
                    "page": r["page"],
                    "chunk": r["chunk"]
                }
                for r in batch
            ]
        )

    print("\nIngestion complete!")
    print(f"PDF pages with text: {len(pages)}")
    print(f"Chunks indexed: {collection.count()}")
    print(f"Database: {DB_DIR}")

if __name__ == "__main__":
    main()
