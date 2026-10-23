
from pathlib import Path
import requests
import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
DB_DIR = BASE / "chroma_db"
COLLECTION_NAME = "prompt_engineering"

# Change this if `ollama list` shows another installed model.
OLLAMA_MODEL = "qwen3:0.6b"
OLLAMA_URL = "http://localhost:11434/api/chat"

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))

try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    raise SystemExit("No indexed documents. Run python ingest.py first.")

def retrieve(question, k=4):
    query_embedding = embedder.encode(
        [question], normalize_embeddings=True
    ).tolist()[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    sources = []
    for text, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        sources.append({
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance
        })

    return sources

def answer_question(question, use_rag=True):
    sources = retrieve(question) if use_rag else []

    context = "\n\n".join(
        f"[Source: {s['source']}, page {s['page']}]\n{s['text']}"
        for s in sources
    )

    if use_rag:
        system_prompt = (
            "You are a careful assistant answering questions about "
            "Prompt Engineering. Answer using only the supplied context. "
            "If the context does not contain the answer, say: "
            "'The provided documents do not contain enough information "
            "to answer this question.' Do not invent citations."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Give a clear, concise answer grounded in the context."
        )
    else:
        system_prompt = (
            "You are a helpful assistant. Answer from your general "
            "knowledge. Do not claim you consulted any documents."
        )
        user_prompt = question

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    answer = response.json()["message"]["content"]

    print("\n" + "=" * 60)
    print("MODE:", "RAG" if use_rag else "MODEL ONLY")
    print("\nANSWER:\n", answer)

    if use_rag:
        print("\nRETRIEVED SOURCES:")
        for s in sources:
            print(
                f"- {s['source']} | page {s['page']} "
                f"| distance={s['distance']:.3f}"
            )
            print("  ", s["text"][:220].replace("\n", " "), "...")

def main():
    print("\nPromptEngineering RAG is ready.")
    print("Type a question, or type 'exit' to quit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        mode = input("Use RAG? (Y/n): ").strip().lower()
        use_rag = mode not in {"n", "no"}

        try:
            answer_question(question, use_rag)
        except requests.exceptions.ConnectionError:
            print("Cannot connect to Ollama. Start the Ollama app first.")
        except requests.exceptions.HTTPError as exc:
            print("Ollama returned an HTTP error:", exc)
            print("Check the model name using: ollama list")
        except Exception as exc:
            print("Error:", exc)

if __name__ == "__main__":
    main()
