import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
from io import BytesIO
import markdown
from weasyprint import HTML
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool, ToolRuntime
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai
import numpy as np
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

load_dotenv()

# --- Configurações Globais ---
GEMINI_EMBEDD = True
COLLECTION_NAME = "Tide"
EMBED_DIM = 768
MODEL_NAME = "gemini-2.5-flash-lite"

env = Environment(loader=FileSystemLoader('templates'))

# --- SINGLETONS (Gerenciadores de Conexão) ---

# Variáveis globais privadas para armazenar as instâncias
_qdrant_instance = None
_embedding_instance = None
_llm_instance = None

def get_qdrant_client():
    """Retorna a instância única do Qdrant Client."""
    global _qdrant_instance
    if _qdrant_instance is None:
        print("[SISTEMA] Iniciando conexão com Qdrant...")
        _qdrant_instance = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
    return _qdrant_instance

def get_embedding_model():
    """Retorna a instância única do modelo de Embedding (Gemini ou Local)."""
    global _embedding_instance
    if _embedding_instance is None:
        print(f"[SISTEMA] Carregando modelo de embedding ({'Gemini' if GEMINI_EMBEDD else 'Local'})...")
        if GEMINI_EMBEDD:
            # Cliente do Google GenAI
            _embedding_instance = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
        else:
            # Modelo SentenceTransformer
            EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
            _embedding_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_instance

def get_llm():
    """Retorna a instância única do LLM."""
    global _llm_instance
    if _llm_instance is None:
        print("[SISTEMA] Iniciando LLM Gemini...")
        _llm_instance = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model=MODEL_NAME,
            temperature=0,
            max_tokens=2048,
            timeout=None,
            max_retries=1,
            transport="rest"            
        )
    return _llm_instance


# --- Funções Auxiliares ---

def normalize(vec):
    v = np.array(vec)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()

def get_embedding(text: str):
    """Gera o embedding usando a instância Singleton."""
    model = get_embedding_model() # <--- Chama o Singleton aqui
    
    if GEMINI_EMBEDD:
        EMBEDDING_MODEL_NAME = "text-embedding-004"
        result = model.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=[text],
            config=genai.types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBED_DIM
            )
        )
        return normalize(result.embeddings[0].values)
    else:
        # Se for SentenceTransformer
        return model.encode(text).tolist()

# --- Ferramentas (Tools) ---

@tool
def retrieve_information(query: str) -> str:
    """Retorna documentos com informacoes confiaveis e relevantes sobre aspectos da menopausa.
    Esta ferramenta é útil para obter informações detalhadas sobre sintomas, tratamentos,
    impacto na saúde mental, dicas de estilo de vida e outros tópicos relacionados à saúde da mulher durante a menopausa.
    Args:
        query (str): A consulta sobre a qual recuperar informações.
        
    Returns:
        str: Documentos informativos relevantes formatados sobre sua consulta.
    """
    
    print(f"[DEBUG] Iniciando busca direta para: {query}")

    try:
        embedding = get_embedding(query)
        client = get_qdrant_client()

        # Busca no Qdrant
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=4 
        )
        
        if not results.points:
            return "⚠️ Nenhum documento relevante encontrado na base de dados."

        # Formatação dos resultados
        formatted_docs = []
        for idx, point in enumerate(results.points, 1):
            texto = point.payload.get('texto', '[Texto não disponível]')
            fonte = point.payload.get('fonte', '[Fonte não disponível]')
            
            doc_str = (
                f"📄 DOCUMENTO {idx}:\n"
                f"{texto}\n\n"
                f"🔗 FONTE: {fonte}\n"
                f"{'-'*80}"
            )
            formatted_docs.append(doc_str)

        final_response = (
            f"\n{'='*80}\n"
            f"📚 DOCUMENTOS RECUPERADOS PARA: '{query}'\n"
            f"{'='*80}\n"
            f"{'\n'.join(formatted_docs)}\n"
            f"{'='*80}\n"
            f"⚠️ IMPORTANTE: Sempre cite a fonte (link) das informações utilizadas.\n"
        )
        
        return final_response

    except Exception as e:
        error_msg = f"[ERROR] Falha na busca vetorial: {str(e)}"
        print(error_msg)
        return "Desculpe, ocorreu um erro técnico ao buscar os documentos."

@tool
def send_pdf(runtime: ToolRuntime) -> str:
    """Envia automaticamente um PDF com o guia personalizado sobre a menopausa para o email do usuário.
    
    IMPORTANTE: Esta ferramenta NÃO requer nenhum parâmetro do usuário. 
    O email e o guia já estão armazenados no sistema a partir das informações coletadas anteriormente.
    Use esta ferramenta quando o usuário solicitar o envio do guia por email.
    
    Returns:
        str: Mensagem indicando o status do envio do PDF (sucesso ou erro).
    """

    user_data = runtime.state.get("user_data", {})
    guide = user_data.get("guide", None)
    email = user_data.get("email", None)
    nome = user_data.get("nome", "Usuária")

    if not guide:
        return "O usuário ainda não gerou o guia. Explique que ele precisa gerar o guia primeiro e pergunte se ele quer gerar o guia agora."
    
    if not email or email == "Não informado":
        return "Não encontrei o email do usuário. Solicite o email antes de enviar."

    try:
       
        remetente = os.getenv("REMETENTE")
        senha = os.getenv("EMAIL_PASSWORD")
        
        print(f"[DEBUG] Iniciando envio de email para: {email}")
        
        # Converter Markdown para HTML
        guide_html = markdown.markdown(guide, extensions=['extra', 'nl2br'])
        
        # Estilização CSS (Mantida igual para consistência visual)
        styled_guide_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                }}
                body {{
                    font-family: 'Arial', 'Helvetica', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 100%;
                }}
                h1 {{
                    color: #d946a6;
                    border-bottom: 3px solid #d946a6;
                    padding-bottom: 10px;
                    margin-top: 20px;
                }}
                h2 {{
                    color: #e879b9;
                    margin-top: 25px;
                    margin-bottom: 15px;
                }}
                h3 {{
                    color: #555;
                }}
                ul, ol {{
                    margin-left: 20px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
                p {{
                    margin-bottom: 12px;
                }}
                strong {{
                    color: #222;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid #ddd;
                    margin: 20px 0;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 10px;
                    border-top: 1px solid #ddd;
                    font-size: 0.9em;
                    color: #666;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            {guide_html}
            <div class="footer">
                <p>Documento gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
            </div>
        </body>
        </html>
        """
        
        # transformar HTML em PDF
        pdf_buffer = BytesIO()
        HTML(string=styled_guide_html).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        
        # Gerar Corpo do Email com Jinja2
        try:
            template = env.get_template('email_template.html')
            corpo_email = template.render(nome=nome)
        except Exception as e:
            print(f"[WARNING] Erro ao carregar template Jinja2: {e}. Usando fallback.")
            corpo_email = f"Olá {nome}, seu guia está em anexo."

        print(f"[DEBUG] PDF gerado com sucesso ({len(pdf_buffer.getvalue())} bytes)")
        
        # metadados do email montar e-mail
        msg = MIMEMultipart()
        msg['Subject'] = '🌸 Seu Guia Personalizado Para Consulta'
        msg['From'] = remetente
        msg['To'] = email
        
        
        msg.attach(MIMEText(corpo_email, 'html', 'utf-8'))
        
        # Anexar PDF
        pdf_attachment = MIMEApplication(pdf_buffer.getvalue(), _subtype='pdf')
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename=f'guia_consulta_{datetime.now().strftime("%Y%m%d")}.pdf')
        msg.attach(pdf_attachment)
        
        # enviar email
        print(f"[DEBUG] Conectando ao servidor SMTP...")
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(remetente, senha)
            server.send_message(msg)
        
        print(f"[DEBUG] Email enviado com sucesso!")
        
        return f"✅ Guia enviado com sucesso para o email {email}! Verifique sua caixa de entrada (e também a pasta de spam, só por precaução). 📧✨"
    
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Erro ao enviar email: {error_msg}")
        return f"❌ Desculpe, houve um erro ao enviar o email: {error_msg}. Por favor, tente novamente mais tarde ou verifique se o email fornecido está correto."

TOOLS_CHAT = [
    retrieve_information,
    send_pdf
]