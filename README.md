# AI PDF Chat

AI PDF Chat is a mini RAG application that lets users upload a PDF and ask questions about its content. The app extracts text, creates custom embeddings, stores vectors in ChromaDB, retrieves relevant context, and generates an answer using a small transformer model built from scratch.

## Features

- Upload and process PDF files.
- Extract text from PDFs.
- Split text into searchable chunks.
- Train custom Word2Vec-style embeddings.
- Store vectors in ChromaDB.
- Retrieve relevant PDF context for user questions.
- Generate answers with a custom transformer LLM.
- Display retrieval and answer-quality metrics in the UI.

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML, CSS, JavaScript, Jinja2
- **PDF Processing**: pypdf
- **ML/DL**: TensorFlow, Keras, NumPy
- **Text Splitting**: LangChain text splitters
- **Vector Database**: ChromaDB

## Project Structure

```text
AIPDFChat/
├── main.py              # FastAPI routes and app setup
├── pipeline.py          # PDF processing, models, retrieval, generation, metrics
├── config.py            # Configuration and hyperparameters
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Web UI
├── uploads/             # Uploaded PDFs
├── models/              # Saved embedding and LLM models
├── tokenizers/          # Saved tokenizers
├── vector_dbs/          # Chunk vector databases
└── chroma_dbs/          # Sentence vector databases
```

## How It Works

```text
PDF upload
   ↓
Text extraction
   ↓
Chunking
   ↓
Custom embedding training
   ↓
Vector storage in ChromaDB
   ↓
Question asked by user
   ↓
Relevant context retrieval
   ↓
Transformer answer generation
   ↓
Answer and metrics displayed
```

## Models Used

### Skip-Gram Embedding Model

A custom Word2Vec-style skip-gram model is trained on the uploaded PDF text. It learns word embeddings that are used to convert questions and PDF chunks into vectors for semantic search.

Optimizer: `Adam`  
Loss: `sparse_categorical_crossentropy`

### Transformer LLM

A small causal transformer model is trained from scratch on the PDF text. It receives the user question and retrieved context, then generates an answer token by token.

Optimizer: `Adam`  
Loss: `sparse_categorical_crossentropy`

## Key Hyperparameters

| Component | Hyperparameter | Value |
|---|---|---:|
| Chunking | `chunk_size` | `1000` |
| Chunking | `chunk_overlap` | `200` |
| Embeddings | `embedding_dim` | `128` |
| Embeddings | `window_size` | `10` |
| Embeddings | `epochs` | `15` |
| Transformer | `sequence_length` | `48` |
| Transformer | `num_heads` | `4` |
| Transformer | `ff_dim` | `256` |
| Transformer | `epochs` | `5` |
| Transformer | `max_new_tokens` | `80` |

## Metrics

The UI displays:

- LLM loss
- LLM accuracy
- LLM perplexity
- Estimated answer accuracy
- Context grounding
- Query coverage
- Answer/context similarity
- Retrieval confidence
- Retrieval distances

These metrics help estimate how well the model performed and how grounded the generated answer is in the retrieved PDF content.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open the app:

```text
http://127.0.0.1:8000
```

## Usage

1. Upload a PDF.
2. Wait for processing and model training to complete.
3. Ask a question about the PDF.
4. View the generated answer, retrieved chunks, and metrics.

## Limitations

- Scanned PDFs are not supported without OCR.
- The transformer is small and trained only on the uploaded PDF.
- Answer quality depends on PDF length and text quality.
- Metrics are estimates because no human reference answer is available.

