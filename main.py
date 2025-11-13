import os
import uuid
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from openai import OpenAI

# ========= CONFIG =========

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "rag_hybrid_collection")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# text-embedding-3-small → 1536 dimensiones, barato y bueno
EMBEDDING_MODEL = "text-embedding-3-small"

if not all([QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY]):
    print(
        "ADVERTENCIA: faltan variables de entorno. "
        "Configura OPENAI_API_KEY, QDRANT_URL y QDRANT_API_KEY en Render."
    )

client_oai = OpenAI(api_key=OPENAI_API_KEY)
client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

app = FastAPI(
    title="RAG híbrido con Qdrant y FastAPI",
    description="Backend RAG híbrido (vector + full-text) para Custom GPT",
    version="1.0.0",
)

# ========= MODELOS Pydantic =========

class IngestItem(BaseModel):
    id: Optional[str] = None
    text: str
    source: Optional[str] = None        # archivo, "chat", etc.
    meta: Optional[dict] = None         # {"fojas": 23, "causa": "...", ...}

class IngestRequest(BaseModel):
    items: List[IngestItem]

class AskRequest(BaseModel):
    question: str
    k: int = 5

class ContextItem(BaseModel):
    text: str
    source: Optional[str] = None
    score: float
    meta: Optional[dict] = None

class AskResponse(BaseModel):
    context: List[ContextItem]

# ========= FUNCIONES AUXILIARES =========

def ensure_collection():
    """
    Crea la colección en Qdrant si no existe, con:
    - vector denso (1536 dims)
    - campo de texto "text" con índice full-text
    """
    if QDRANT_URL is None:
        return

    try:
        collections = client_qdrant.get_collections()
        names = [c.name for c in collections.collections]
    except Exception as e:
        print(f"Error listando colecciones Qdrant: {e}")
        return

    if QDRANT_COLLECTION not in names:
        print(f"Creando colección {QDRANT_COLLECTION}...")
        client_qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=1536,
                distance=qmodels.Distance.COSINE,
            ),
        )

    # Crear índice full-text sobre el campo "text"
    try:
        client_qdrant.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="text",
            field_schema=qmodels.TextIndexParams(
                type="text",
                tokenizer=qmodels.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )
        print("Índice full-text creado para campo 'text'.")
    except Exception as e:
        # Si ya existe, Qdrant puede lanzar error; lo ignoramos
        print(f"Índice full-text puede que ya exista: {e}")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Llama a OpenAI para generar embeddings de una lista de textos.
    """
    resp = client_oai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [d.embedding for d in resp.data]

def rrf_fusion(
    dense_hits: List[qmodels.ScoredPoint],
    text_hits: List[qmodels.ScoredPoint],
    k_rrf: int = 60,
    top_k: int = 5,
) -> List[qmodels.ScoredPoint]:
    """
    Fusiona resultados dense + dense+texto usando Reciprocal Rank Fusion.
    Cada lista viene ordenada por score descendente.
    Score_RRF = Σ 1 / (k_rrf + rank).
    """
    scores: Dict[str, float] = {}
    objs: Dict[str, qmodels.ScoredPoint] = {}

    # asignar RRF por dense
    for rank, hit in enumerate(dense_hits):
        pid = str(hit.id)
        scores.setdefault(pid, 0.0)
        scores[pid] += 1.0 / (k_rrf + rank + 1)
        objs[pid] = hit

    # asignar RRF por text-filtered
    for rank, hit in enumerate(text_hits):
        pid = str(hit.id)
        scores.setdefault(pid, 0.0)
        scores[pid] += 1.0 / (k_rrf + rank + 1)
        objs[pid] = hit

    # ordenar por score combinado
    ranked_ids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)
    fused = [objs[pid] for pid in ranked_ids[:top_k]]
    return fused

# ========= EVENTO DE INICIO =========

@app.on_event("startup")
def startup_event():
    ensure_collection()

# ========= ENDPOINTS =========

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=dict)
def ingest(data: IngestRequest):
    """
    Guarda textos (docs, resúmenes de chat, etc.) en Qdrant.
    """
    if not data.items:
        raise HTTPException(status_code=400, detail="No hay items para ingerir")

    texts = [item.text for item in data.items]
    vectors = embed_texts(texts)

    points = []
    for item, vec in zip(data.items, vectors):
        point_id = item.id or str(uuid.uuid4())
        payload = {
            "text": item.text,
        }
        if item.source:
            payload["source"] = item.source
        if item.meta:
            payload["meta"] = item.meta

        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vec,
                payload=payload,
            )
        )

    client_qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    return {"inserted": len(points)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    RAG híbrido:
    - búsqueda densa
    - búsqueda densa + filtro full-text (MatchText sobre 'text')
    - fusión RRF de ambos resultados
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Pregunta vacía")

    # 1) embedding de la pregunta
    q_vec = embed_texts([question])[0]

    # 2) búsqueda densa pura
    try:
        dense_hits = client_qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=max(req.k, 10),   # un poco más para que la fusión tenga material
            with_payload=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda dense: {e}")

    # 3) búsqueda densa con filtro full-text
    #    usamos MatchText sobre 'text' con la pregunta completa.
    #    Si la pregunta tiene varias palabras, Qdrant exige que estén todas.
    text_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="text",
                match=qmodels.MatchText(text=question),
            )
        ]
    )

    try:
        text_hits = client_qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=max(req.k, 10),
            with_payload=True,
            query_filter=text_filter,
        )
    except Exception as e:
        # si falla (ej. sin índice), degradamos a lista vacía
        print(f"Error en búsqueda full-text: {e}")
        text_hits = []

    # 4) fusión RRF de dense + text
    fused_hits = rrf_fusion(
        dense_hits=dense_hits,
        text_hits=text_hits,
        k_rrf=60,
        top_k=req.k,
    )

    context_items: List[ContextItem] = []
    for hit in fused_hits:
        payload = hit.payload or {}
        context_items.append(
            ContextItem(
                text=payload.get("text", ""),
                source=payload.get("source"),
                score=float(hit.score),
                meta=payload.get("meta"),
            )
        )

    return AskResponse(context=context_items)
