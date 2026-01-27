import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

# ==== CONFIGURAÇÕES ====
CHUNKS_PATH = Path("index/files/doc_chunks.jsonl")
COLLECTION_NAME = "Tide"
MODEL_NAME = "all-MiniLM-L6-v2"

# Substitua pelos seus dados do Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("🔹 Carregando chunks...")
chunks = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"✅ {len(chunks)} chunks carregados.")

print(f"🔹 Carregando modelo '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

print("🔹 Gerando embeddings (pode levar alguns minutos)...")
texts = [c["chunk_text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

print(f"✅ Embeddings gerados: {len(embeddings)} vetores de dimensão {embeddings.shape[1]}")

print("🔹 Conectando ao Qdrant Cloud...")
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# ==== 5. Cria (ou recria) a coleção ====
print(f"🔹 Criando a coleção '{COLLECTION_NAME}'...")
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=embeddings.shape[1],
        distance=models.Distance.COSINE
    )
)

print("🔹 Inserindo pontos no Qdrant Cloud...")
BATCH_SIZE = 64
points = []

for i, (chunk, emb) in enumerate(tqdm(zip(chunks, embeddings), total=len(chunks))):
    # Tenta obter o link de diferentes formatos
    fonte = (
        chunk.get("source")
        or chunk.get("metadata", {}).get("source")
    )

    payload = {
        "id_original": chunk.get("original_id"),
        "indice_de_blocos": chunk.get("chunk_index"),
        "texto": chunk.get("chunk_text"),
        "fonte": fonte,
    }

    points.append(models.PointStruct(id=i, vector=emb.tolist(), payload=payload))

    # Envia em lotes
    if len(points) >= BATCH_SIZE:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        points = []

# Envia os pontos restantes
if points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print("✅ Indexação concluída com sucesso no Qdrant Cloud!")

# ==== 7. Teste de busca ====
query = "Quais os sintomas da menopausa?"
query_vector = model.encode(query).tolist()

print(f"\nRealizando busca no Qdrant Cloud...")
search_results = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=3
)

print("\nResultados mais semelhantes:")
for r in search_results:
    print("-" * 80)
    print(f"Score: {r.score:.4f}")
    print(r.payload["text"][:500])
    print()
