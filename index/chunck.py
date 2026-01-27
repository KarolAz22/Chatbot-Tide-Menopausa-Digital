import json
from pathlib import Path

#OBS: Se você já tem o arquivo doc_chuncks não precisa executar este arquivo

# ==== CONFIGURAÇÕES ====
INPUT_PATH = Path("index/files/doc_clean_unstructured.jsonl")
OUTPUT_PATH = Path("index/files/doc_chunks.jsonl")

MAX_CHARS = 1000
OVERLAP = 200

# ==== Função para carregar JSONL ====
def load_jsonl(path: Path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                obj = {"text": line}
            obj["_line_index"] = i
            docs.append(obj)
    return docs

# ==== Detecta automaticamente o campo de texto principal ====
def get_main_text_field(doc):
    candidates = ["text", "content", "body", "document", "raw", "plain_text", "page_content"]
    for c in candidates:
        if c in doc and isinstance(doc[c], str) and doc[c].strip():
            return c
    string_fields = {k: v for k, v in doc.items() if isinstance(v, str)}
    if not string_fields:
        return None
    longest_field = max(string_fields.keys(), key=lambda k: len(string_fields[k]))
    return longest_field

# ==== Função para quebrar o texto em chunks ====
def make_chunks_from_text(text, max_chars=1000, overlap=200):
    if not text:
        return []
    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = start + max_chars
        if end >= n:
            end = n
            chunk_piece = text[start:end].strip()
            if chunk_piece:
                chunks.append((start, end, chunk_piece))
            break
        window_start = max(start, end - 200)
        slice_candidate = text[window_start:end]
        last_period = max(slice_candidate.rfind("."), slice_candidate.rfind("!"), slice_candidate.rfind("?"))
        if last_period != -1:
            end = window_start + last_period + 1
        chunk_piece = text[start:end].strip()
        if not chunk_piece:
            end = start + max_chars
            chunk_piece = text[start:end].strip()
        chunks.append((start, end, chunk_piece))
        start = max(start + max_chars - overlap, end)
    return chunks

# ==== Carrega documentos ====
docs = load_jsonl(INPUT_PATH)
if not docs:
    raise RuntimeError("Nenhum documento encontrado no arquivo de entrada!")

text_field = None
for doc in docs:
    tf = get_main_text_field(doc)
    if tf:
        text_field = tf
        break

if text_field is None:
    raise RuntimeError("Não foi possível identificar o campo de texto principal!")

print(f"Campo de texto detectado: {text_field}")

# ==== Cria os chunks ====
chunks_out = []
for doc_idx, doc in enumerate(docs):
    original_id = doc.get("id") or f"line_{doc.get('_line_index', doc_idx)}"
    text_value = doc.get(text_field, "")

    # tenta encontrar o link (source/url/link) em vários lugares
    link_value = None
    possible_link_keys = ["url", "link", "source", "href"]
    
    # procura no nível principal
    for key in possible_link_keys:
        if key in doc:
            link_value = doc[key]
            break
    
    # se não encontrou, procura dentro de 'metadata'
    if not link_value and isinstance(doc.get("metadata"), dict):
        metadata = doc["metadata"]
        for key in possible_link_keys:
            if key in metadata:
                link_value = metadata[key]
                break

    doc_chunks = make_chunks_from_text(text_value, max_chars=MAX_CHARS, overlap=OVERLAP)

    for i, (start, end, piece) in enumerate(doc_chunks):
        chunk_obj = {
            "original_id": original_id,
            "source_line_index": doc.get("_line_index"),
            "chunk_index": i,
            "start_char": start,
            "end_char": end,
            "chunk_length": len(piece),
            "chunk_text": piece,
        }

        # adiciona metadados úteis
        if link_value:
            chunk_obj["source"] = link_value
        if "title" in doc:
            chunk_obj["title"] = doc["title"]
        if "filename" in doc:
            chunk_obj["filename"] = doc["filename"]

        chunks_out.append(chunk_obj)

print(f"✅ Total de {len(chunks_out)} chunks gerados a partir de {len(docs)} documentos.")

# ==== Salva no novo JSONL ====
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for c in chunks_out:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"✅ Arquivo salvo em: {OUTPUT_PATH}")
