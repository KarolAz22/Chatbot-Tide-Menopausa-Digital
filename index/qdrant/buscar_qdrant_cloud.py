from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os

# ==== CONFIGURAÇÕES ====
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "Tide"

# Substitua com os dados do seu Qdrant Cloud
# Substitua pelos seus dados do Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("🔹 Conectando ao Qdrant Cloud...")
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

print(f"🔹 Carregando modelo '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

# ==== Define a consulta ====
query = input("Digite sua pergunta: ")

# ==== Gera embedding da pergunta ====
query_vector = model.encode(query).tolist()

print("\nBuscando na coleção...")
search_results = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=5  # Retorna os 5 resultados mais relevantes
)

print("Resultados mais semelhantes:")
for r in search_results:
    print("-" * 80)
    print(f"Score: {r.score:.4f}")
    print(r.payload["texto"][:500])
    print()

# Busca os primeiros 3 pontos inseridos
results = qdrant.scroll(
    collection_name=COLLECTION_NAME,
    limit=3,
    with_payload=True,     # <- mostra os metadados
    with_vectors=False     # <- não precisa mostrar o vetor
)

print("=== Verificação de metadados ===")
for point in results[0]:
    print("-" * 80)
    print(f"ID: {point.id}")
    print("Payload:")
    for key, value in point.payload.items():
        print(f"  {key}: {value}")