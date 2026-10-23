# PromptEngineering-RAG
### Retrieval-Augmented Generation (RAG) Using Personal Documents

## 📌 Project Overview

PromptEngineering-RAG is a Retrieval-Augmented Generation (RAG) system that answers user questions using information retrieved from a custom collection of documents. Instead of relying only on an LLM's pretrained knowledge, the system retrieves relevant document chunks from a local vector database and provides them as context to the language model.

The project demonstrates document processing, text chunking, semantic search, embedding generation, vector storage, prompt construction, and grounded response generation.

## 🎯 Objectives

- Build a working RAG pipeline using real PDF documents.
- Extract and divide document content into manageable chunks.
- Generate semantic embeddings for the document chunks.
- Store embeddings and metadata in ChromaDB.
- Retrieve relevant context for a user's question.
- Generate answers using retrieved context and a locally running LLM.
- Compare responses generated with retrieval against responses generated without retrieval.

## 🛠️ Technologies Used

- **Python** — core implementation
- **ChromaDB** — local vector database
- **Sentence Transformers** — text embedding generation
- **PyPDF** — PDF text extraction
- **Ollama** — local LLM inference
- **Requests** — communication with the Ollama API
- **VS Code** — development environment

## 🏗️ System Architecture

1. **Document Ingestion:** Load PDF documents from the `data/` directory.
2. **Text Extraction:** Extract readable text and retain document metadata, including source filename and page number.
3. **Text Chunking:** Split extracted text into smaller, overlapping chunks to support more focused retrieval.
4. **Embedding Generation:** Convert chunks into vector representations using Sentence Transformers.
5. **Vector Storage:** Store embeddings, text chunks, and metadata in ChromaDB.
6. **Semantic Retrieval:** Retrieve the most relevant chunks for a user's query.
7. **Prompt Construction:** Add the retrieved content and user question to the LLM prompt.
8. **Response Generation:** Use Ollama to generate a response based on the supplied context.
9. **Evaluation:** Compare RAG-enabled responses with model-only responses.

## 📁 Project Structure

```text
PromptEngineering-RAG/
├── data/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── chroma_db/
├── venv/
├── ingest.py
├── query.py
├── transcribe_video.py        # If using video transcription
├── README.md
└── video.mp4                  # Optional; keep out of Git if large
```

*Note: The PDF filenames above are examples. Replace them with the actual filenames in your project. The `transcribe_video.py` file is optional if video transcription is part of your implementation.*

## ⚙️ Installation and Setup

### Prerequisites

- Python installed
- Ollama installed and running
- A local Ollama model downloaded
- PDF documents available for ingestion

### 1. Clone the repository

```bash
git clone <YOUR_PUBLIC_GITHUB_REPOSITORY_URL>
cd PromptEngineering-RAG
```

### 2. Create and activate a virtual environment

For Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install chromadb sentence-transformers pypdf requests
```

If video transcription is included, install its additional dependency:

```powershell
pip install faster-whisper
```

### 4. Configure Ollama

Ensure Ollama is installed and running. Download a model if necessary:

```powershell
ollama pull qwen3:0.6b
```

Check the available models:

```powershell
ollama list
```

**Important:** The model name in `query.py` must match a model installed on your system. If you use a different model, update the `OLLAMA_MODEL` variable accordingly.

### 5. Add documents

Place your PDF documents in the `data/` directory. Use a meaningful collection of real content rather than a small sample text file.

### 6. Ingest and index the documents

```powershell
python .\ingest.py
```

This processes the documents, generates embeddings, and stores the resulting chunks in ChromaDB.

### 7. Start the RAG application

```powershell
python .\query.py
```

Enter questions when prompted. Use the RAG option to retrieve document context, and the model-only option to compare responses without retrieval.

## 🔍 Chunking and Retrieval

The ingestion pipeline extracts document text and divides it into smaller chunks. Overlapping chunks help preserve context across chunk boundaries.

The system generates embeddings for each chunk and stores them in ChromaDB alongside source metadata. At query time, the question is embedded and compared against the stored vectors to retrieve the most relevant content.

**Chunking note:** The current implementation uses character-based chunking, which only approximates token-based chunk sizes. If the task strictly requires chunks of 300–800 tokens, verify and adjust the chunking logic before submission.

## 🧪 Testing and Evaluation

The system should be evaluated using questions whose answers can be verified against the ingested documents.

### Test 1: Document-specific question

Ask a question whose answer is explicitly present in one of the PDFs.

**Expected result:** The system retrieves relevant content and generates an answer supported by that content.

### Test 2: Retrieval relevance

Inspect the retrieved chunks, their source filenames, and page references.

**Expected result:** The retrieved content is relevant to the question and provides evidence for the generated answer.

### Test 3: RAG versus model-only comparison

Ask the same document-specific question in both modes.

| Evaluation | RAG-enabled mode | Model-only mode |
|---|---|---|
| Retrieved document context | Included | Not included |
| Access to document-specific information | Available through retrieval | Not supplied by the documents |
| Answer grounding | Can be checked against retrieved passages | May rely on pretrained knowledge |
| Source verification | Inspect retrieved source and page metadata | No retrieved document source |

### Comparison Evidence

Add an actual test example here before submission.

**Question:** [Insert a question answerable only from your documents]

**RAG answer:** [Paste the answer returned by your system]

**Retrieved evidence:** [Paste the relevant passage, source filename, and page number]

**Model-only answer:** [Paste the answer returned without retrieval]

**Observation:** [Explain whether the RAG response used the document-specific information and whether the model-only response lacked that information or answered incorrectly.]

Do not assume that the model-only response will always be wrong. Record the actual output from both runs.

## 🎥 Optional Video Transcription

If video transcription is included, the project can use Faster-Whisper to convert spoken content from an MP4 file into a text transcript. The transcript can then be added to the document ingestion pipeline, provided the ingestion code supports the transcript format.

## 🚀 Future Improvements

- Token-aware chunking and configurable overlap
- Support for additional document formats
- Improved retrieval ranking and relevance evaluation
- Source citations in generated answers
- A web-based chat interface
- Automated evaluation of answer faithfulness and retrieval quality

## 🔐 Repository and Data Notes

- Do not commit the virtual environment (`venv/`).
- Avoid uploading large video files unless required.
- Check whether your PDFs can legally be shared publicly before committing them.
- Exclude generated files and local databases if they are unnecessary for the submission.
- Include only implementation details and test results that have been verified.

## 📸 Experiment Evidence

Screenshots documenting the RAG pipeline and evaluation are available in
the [`photos/`](photos/) directory.

- `ingestion_success.png` — successful document ingestion and indexing.
- `rag_answer.png` — answer generated with retrieved document context.
- `model_only_answer.png` — answer generated without retrieval.
- `retrieval_evidence.png` — retrieved chunks and source/page metadata.

The comparison demonstrates how document retrieval affects answers to
questions that depend on the ingested content.

## 👩‍💻 Author

**Anjali Wadekar**  
Final-Year B.Tech Computer Science and Engineering Student

---

*This project was developed as a practical implementation of Retrieval-Augmented Generation and prompt engineering concepts.*
