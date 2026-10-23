
from pathlib import Path
import chromadb
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "prompt_engineering"
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading ChromaDB...")
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(COLLECTION_NAME)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

result = collection.get(include=["documents", "metadatas"])
documents = result["documents"] or []
metadatas = result["metadatas"] or []

if not documents:
    print("No chunks found. Run python .\\ingest.py first.")
    raise SystemExit(1)

token_counts = [
    len(tokenizer.encode(doc, add_special_tokens=False))
    for doc in documents
]

print(f"\nTotal chunks: {len(documents)}")
print(f"Smallest chunk: {min(token_counts)} tokens")
print(f"Largest chunk: {max(token_counts)} tokens")
print(f"Average chunk: {sum(token_counts) / len(token_counts):.1f} tokens")

within_range = sum(300 <= count <= 800 for count in token_counts)
print(
    f"Chunks within 300-800 tokens: "
    f"{within_range}/{len(token_counts)} "
    f"({within_range / len(token_counts) * 100:.1f}%)"
)

print("\n--- First 10 chunks ---")
for i, (doc, metadata, count) in enumerate(
    zip(documents[:10], metadatas[:10], token_counts[:10]), start=1
):
    print(f"\nChunk {i}: {count} tokens")
    print(f"Source: {metadata.get('source', 'Unknown')}")
    print(f"Page: {metadata.get('page', 'Unknown')}")
    print(doc[:250].replace("\n", " "))
