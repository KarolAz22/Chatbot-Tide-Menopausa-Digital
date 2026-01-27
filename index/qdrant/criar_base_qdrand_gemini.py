import json
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
load_dotenv()


# ==== CONFIGURAÇÕES ====

CHUNKS_PATH = Path("index/files/doc_chunks.jsonl")
COLLECTION_NAME = "Tide"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

EMBED_DIM = 768


# ==== FUNÇÃO DE NORMALIZAÇÃO ====

def normalize(vec):
    v = np.array(vec)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()


# ==== batching ====

def batch(iterable, n=100):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


# ==== 1. Carregar chunks ====

print("🔹 Carregando chunks...")
chunks = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"✅ {len(chunks)} chunks carregados.")


# ==== 2. Inicializar Gemini ====

print("🔹 Inicializando cliente Gemini...")
client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)


# ==== 3. Gerar embeddings ====

print("🔹 Gerando embeddings com Gemini...")

texts = [c["chunk_text"] for c in chunks]
all_embeddings = []

for text_batch in tqdm(list(batch(texts, 100))):

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text_batch,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBED_DIM
        )
    )

    for emb in response.embeddings:
        all_embeddings.append(normalize(emb.values))

embeddings = np.array(all_embeddings)

print(f"✅ Total de embeddings gerados: {len(embeddings)} vetores de {EMBED_DIM} dimensões.")


# ==== 4. Conectar ao Qdrant ====

print("🔹 Conectando ao Qdrant Cloud...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ==== 5. Criar/Recriar coleção ====

print(f"🔹 Criando ou atualizando coleção '{COLLECTION_NAME}'...")

qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=EMBED_DIM,  # agora 768
        distance=models.Distance.COSINE
    )
)


# ==== 6. Inserir documentos ====

print("🔹 Inserindo pontos no Qdrant...")

BATCH_SIZE = 64
buffer_points = []

for i, (chunk, emb) in enumerate(tqdm(zip(chunks, embeddings), total=len(chunks))):

    fonte = chunk.get("source") or chunk.get("metadata", {}).get("source")

    payload = {
        "id_original": chunk.get("original_id"),
        "indice_de_blocos": chunk.get("chunk_index"),
        "texto": chunk.get("chunk_text"),
        "fonte": fonte,
    }

    buffer_points.append(
        models.PointStruct(
            id=i,
            vector=emb,
            payload=payload
        )
    )

    if len(buffer_points) >= BATCH_SIZE:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=buffer_points)
        buffer_points = []

if buffer_points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=buffer_points)

print("Indexação concluída com sucesso no Qdrant!")
