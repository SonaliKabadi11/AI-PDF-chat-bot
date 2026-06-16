"""
FastAPI application for PDF Semantic Search.

Endpoints
---------
GET  /                  — Serve the UI (Jinja2 template)
POST /upload            — Accept a PDF, run full ingestion pipeline
POST /query             — Run a semantic search against an ingested PDF
GET  /status/{session}  — Poll ingestion progress
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from config import PipelineConfig
from pipeline import (
    evaluate_answer_metrics,
    extractive_fallback_answer,
    generate_answer,
    get_embedding_weights,
    ingest_pdf,
    load_llm_model,
    load_model_from_path,
    load_tokenizer,
    search_pdf,
    should_use_extractive_answer,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------




app = FastAPI(title="PDF Semantic Search", version="1.0.0")
# templates = Jinja2Templates(directory="templates")
BASE_DIR = Path(__file__).resolve().parent
print("Current file:", BASE_DIR)
print("Template exists:",
      (BASE_DIR / "templates" / "index.html").exists())

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)
cfg = PipelineConfig()

# Ensure required directories exist
for d in [cfg.upload_dir, cfg.vector_db_base, cfg.chroma_db_base,
          cfg.model_save_dir, cfg.llm_save_dir, cfg.tokenizer_save_dir]:
    Path(d).mkdir(parents=True, exist_ok=True)

# In-memory session store  {session_id: {"status": ..., "stats": ..., "filename": ...}}
sessions: Dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Background ingestion worker
# ---------------------------------------------------------------------------

def _run_ingestion(pdf_path: Path, session_id: str, filename: str) -> None:
    sessions[session_id]["status"] = "processing"
    try:
        stats = ingest_pdf(pdf_path, session_id, cfg)
        sessions[session_id].update({"status": "ready", "stats": stats, "filename": filename})
        logger.info("Session %s ready — %s", session_id, stats)
    except Exception as exc:
        logger.exception("Ingestion failed for session %s", session_id)
        sessions[session_id].update({"status": "error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})
    # return {"status": "working"}


@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    session_id = uuid.uuid4().hex
    pdf_path = cfg.upload_dir / f"{session_id}.pdf"

    # Save the uploaded file
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    pdf_path.write_bytes(contents)

    sessions[session_id] = {"status": "queued", "filename": file.filename, "stats": None}

    # Run ingestion in a thread so the HTTP response returns immediately
    background_tasks.add_task(_run_ingestion, pdf_path, session_id, file.filename)

    return JSONResponse({"session_id": session_id, "filename": file.filename})


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return JSONResponse(sessions[session_id])


@app.post("/query")
async def query(session_id: str = Form(...), query_text: str = Form(...), top_n: int = Form(5)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    if session["status"] != "ready":
        raise HTTPException(status_code=400, detail=f"PDF not ready yet. Status: {session['status']}")

    if not query_text.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        model = load_model_from_path(cfg.model_save_dir / f"{session_id}.keras")
        tokenizer = load_tokenizer(cfg.tokenizer_save_dir / f"{session_id}.pickle")
        vocab_size = len(tokenizer.word_index) + 1
        weights = get_embedding_weights(model, cfg.embedding.layer_name)

        results = search_pdf(
            query_text, tokenizer, vocab_size, weights,
            cfg.embedding.embedding_dim,
            cfg.chroma_db_base / session_id,
            cfg.sentence_collection_name,
            top_n=top_n,
        )
        llm_path = cfg.llm_save_dir / f"{session_id}.keras"
        if not llm_path.exists():
            raise ValueError(
                "This PDF session does not have a trained transformer LLM. "
                "Upload and process the PDF again to train the answer generator."
            )
        llm_model = load_llm_model(llm_path)
        answer, generation_info = generate_answer(query_text, results, llm_model, tokenizer, cfg.llm)
        metrics = evaluate_answer_metrics(
            query_text,
            answer,
            results,
            tokenizer,
            vocab_size,
            weights,
            cfg.embedding.embedding_dim,
        )
        metrics.update(generation_info)
        use_extractive, reason = should_use_extractive_answer(metrics, cfg.llm)
        if use_extractive and not generation_info.get("fallback_used"):
            answer = extractive_fallback_answer(query_text, results)
            metrics = evaluate_answer_metrics(
                query_text,
                answer,
                results,
                tokenizer,
                vocab_size,
                weights,
                cfg.embedding.embedding_dim,
            )
            metrics.update(
                {
                    "answer_source": "extractive",
                    "fallback_used": True,
                    "fallback_reason": reason,
                    "generated_token_count": generation_info.get("generated_token_count", 0),
                    "query_type": generation_info.get("query_type", "general"),
                }
            )
        return JSONResponse(
            {
                "answer": answer,
                "metrics": metrics,
                "results": results,
                "query": query_text,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="Search failed. See server logs.")
