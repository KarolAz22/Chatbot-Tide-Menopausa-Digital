from google import genai
from google.genai import types
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv()

# ==== CONFIGURAÇÕES ====
COLLECTION_NAME = "Tide"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBED_DIM = 768

print("🔹 Conectando ao Qdrant Cloud...")
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

print("🔹 Inicializando Gemini...")
client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)

# ==== Pergunta ====
query = input("Digite sua pergunta: ")

# ==== Embedding da pergunta ====
response = client.models.embed_content(
    model="text-embedding-004",
    contents=[query],
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=EMBED_DIM
    )
)

query_vector = response.embeddings[0].values

# ==== Busca no Qdrant ====
print("\n🔍 Buscando no Qdrant...")

result = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5,
    with_payload=True
)

search_results = result.points

# ==== Exibir resultados ====
print("\n=== Resultados mais relevantes ===")
for r in search_results:
    print("-" * 80)
    print(f"Score: {r.score:.4f}")
    print(r.payload.get("texto", "")[:500])
    print()
