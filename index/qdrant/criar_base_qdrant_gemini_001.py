import json
import time
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
# NOVO: Arquivo que vai salvar o seu progresso localmente
BACKUP_PATH = Path("index/files/embeddings_backup.jsonl") 
COLLECTION_NAME = "Tide"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

EMBED_DIM = 3072 


# ==== FUNÇÃO DE NORMALIZAÇÃO ====

def normalize(vec):
    v = np.array(vec)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()


# ==== 1. Carregar todos os chunks originais ====

print("🔹 Carregando chunks...")
chunks = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"✅ {len(chunks)} chunks originais encontrados na base.")


# ==== 2. Verificar Progresso no Backup ====

processed_data = []
if BACKUP_PATH.exists():
    with open(BACKUP_PATH, "r", encoding="utf-8") as f:
        for line in f:
            processed_data.append(json.loads(line))
    print(f"🔄 Retomando de backup: {len(processed_data)} embeddings já estão salvos!")
else:
    print("🆕 Nenhum backup encontrado. Iniciando do zero.")

# Separa apenas os chunks que ainda não foram processados
chunks_to_process = chunks[len(processed_data):]


# ==== 3. Gerar embeddings (Apenas do que falta) ====

if len(chunks_to_process) > 0:
    print("🔹 Inicializando cliente Gemini...")
    client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)
    
    print(f"🔹 Gerando embeddings para os {len(chunks_to_process)} chunks restantes...")
    
    # Abre o arquivo de backup em modo 'append' (adiciona no final sem apagar o que já tem)
    with open(BACKUP_PATH, "a", encoding="utf-8") as f_out:
        
        texts = [c["chunk_text"] for c in chunks_to_process]
        
        # Processando em lotes de 20
        for i in tqdm(range(0, len(texts), 20)):
            text_batch = texts[i:i+20]
            chunk_batch = chunks_to_process[i:i+20]
            
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text_batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        title="Base de Conhecimento Tide Menopausa"
                    )
                )

                # Salva cada resultado imediatamente no arquivo de backup e na memória
                for chunk, emb in zip(chunk_batch, response.embeddings):
                    vec = normalize(emb.values)
                    record = {"chunk": chunk, "vector": vec}
                    
                    # Salva no disco
                    f_out.write(json.dumps(record) + "\n")
                    # Mantém na memória para o passo do Qdrant
                    processed_data.append(record)
                    
                time.sleep(15) # Pausa de segurança da API
                
            except Exception as e:
                print("\n❌ Erro ou limite da API atingido. O seu progresso está salvo até o lote atual.")
                print(f"Detalhe do erro: {e}")
                print("Pare o script e tente rodar novamente amanhã para continuar de onde parou.")
                exit() # Interrompe a execução com segurança
                
else:
    print("✅ Todos os embeddings já foram gerados e estão prontos no backup local!")


# ==== 4. Conectar e Enviar para o Qdrant ====
# Só chega aqui se não tiver dado erro na API de embeddings

print("🔹 Conectando ao Qdrant Cloud...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


print(f"🔹 Recriando coleção '{COLLECTION_NAME}' com {EMBED_DIM} dimensões...")
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=EMBED_DIM, 
        distance=models.Distance.COSINE
    )
)

print("🔹 Inserindo pontos finais no Qdrant...")
BATCH_SIZE = 64
buffer_points = []

for i, data in enumerate(tqdm(processed_data)):
    chunk = data["chunk"]
    emb = data["vector"]
    
    fonte = chunk.get("source") or chunk.get("metadata", {}).get("source")

    payload = {
        "id_original": chunk.get("original_id"),
        "indice_de_blocos": chunk.get("chunk_index"),
        "texto": chunk.get("chunk_text"),
        "fonte": fonte,
    }

    buffer_points.append(
        models.PointStruct(id=i, vector=emb, payload=payload)
    )

    if len(buffer_points) >= BATCH_SIZE:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=buffer_points)
        buffer_points = []

if buffer_points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=buffer_points)

print("🎉 Sucesso absoluto! A base de dados da Tide está online e vetorizada no Qdrant.")

'''import json
import time
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Carrega as variáveis do .env (URLs e chaves de API)
load_dotenv()


# ==== CONFIGURAÇÕES ====

CHUNKS_PATH = Path("index/files/doc_chunks.jsonl")
COLLECTION_NAME = "Tide"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

# A nova dimensão nativa do modelo gemini-embedding-001
EMBED_DIM = 3072 


# ==== FUNÇÃO DE NORMALIZAÇÃO ====

def normalize(vec):
    v = np.array(vec)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()


# ==== FUNÇÃO DE BATCHING (LOTES) ====

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

print("🔹 Gerando embeddings com Gemini (Lotes de 20 com pausa de 15s para evitar bloqueio da API)...")

texts = [c["chunk_text"] for c in chunks]
all_embeddings = []

# Processando de 20 em 20 textos
for text_batch in tqdm(list(batch(texts, 20))):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text_batch,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            title="Base de Conhecimento Tide Menopausa"
        )
    )

    for emb in response.embeddings:
        all_embeddings.append(normalize(emb.values))
        
    # Pausa de 15 segundos: garante no máximo 80 chamadas por minuto (limite é 100)
    time.sleep(15) 

embeddings = np.array(all_embeddings)

print(f"✅ Total de embeddings gerados: {len(embeddings)} vetores de {EMBED_DIM} dimensões.")


# ==== 4. Conectar ao Qdrant ====

print("🔹 Conectando ao Qdrant Cloud...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ==== 5. Criar/Recriar coleção ====

print(f"🔹 Criando/Atualizando coleção '{COLLECTION_NAME}' com {EMBED_DIM} dimensões...")

qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=EMBED_DIM, 
        distance=models.Distance.COSINE
    )
)


# ==== 6. Inserir documentos no banco ====

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

    # Quando o buffer atinge 64, envia para o Qdrant
    if len(buffer_points) >= BATCH_SIZE:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=buffer_points)
        buffer_points = []

# Envia o restante que sobrou no buffer
if buffer_points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=buffer_points)

print("✅ Indexação concluída com sucesso no Qdrant! O banco de dados da Tide está pronto.")'''