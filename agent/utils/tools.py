import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
import markdown # transformar markdown em html
from weasyprint import HTML # transformar HTML em PDF
from io import BytesIO # BytesIO para manipulação de arquivos em memória
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig, ActionRequest
from qdrant_client import QdrantClient
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from google import genai
import numpy as np

load_dotenv()

GEMINI_EMBEDD = True
COLLECTION_NAME = "Tide"
EMBED_DIM = 768

if GEMINI_EMBEDD:
    EMBEDDING_MODEL_NAME = "text-embedding-004"
    embedding_model = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
else:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def normalize(vec):
    v = np.array(vec)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

MODEL_NAME = "gemini-2.5-flash-lite"


llm_ =  ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model=MODEL_NAME,
    temperature=0,
    max_tokens=2048,
    timeout=None,
    max_retries=2,            
)

class Subqueries(BaseModel):
    subquery1: str
    subquery2: str
    subquery3: str

@tool
def retrieve_information(query: str) -> str:
    """Retorna documentos com informacoes confiaveis e relevantes sobre aspectos da menopausa.
    Esta ferramenta é útil para obter informações detalhadas sobre sintomas, tratamentos,
    impacto na saúde mental, dicas de estilo de vida e outros tópicos relacionados à saúde da mulher durante a menopausa.
    Args:
        query (str): A consulta sobre a qual recuperar informações.
    Returns:
        str: Documentos informativos relevantes sobre a consulta.

    """
    
    subquery_prompt = f"""Você é um especialista em reformular consultas para sistemas de busca semântica sobre menopausa e saúde da mulher.

        Dada a consulta original abaixo, crie TRÊS subconsultas distintas que COMPLEMENTEM e EXPANDAM a busca original. 

        ESTRATÉGIAS para criar boas subconsultas:
        1. **Aspecto médico/científico**: Foque nos mecanismos biológicos, hormônios, processos fisiológicos
        2. **Aspecto prático/tratamento**: Foque em tratamentos, terapias, medicamentos, alternativas
        3. **Aspecto específico/relacionado**: Foque em consequências, sintomas relacionados, impactos na vida

        REGRAS IMPORTANTES:
        - Cada subconsulta deve abordar um ÂNGULO DIFERENTE do tema
        - Use termos médicos e sinônimos relevantes
        - Seja específico e direto
        - NÃO repita a consulta original
        - Pense em aspectos que um médico consideraria relacionados

        CONSULTA ORIGINAL: {query}

        Gere as três subconsultas complementares:
        
        """
            
    response = llm_.with_structured_output(Subqueries).invoke([
       HumanMessage(content=subquery_prompt)
    ])

    print(f"[DEBUG] Query original: {query}")
    print(f"[DEBUG] Subquery 1: {response.subquery1}")
    print(f"[DEBUG] Subquery 2: {response.subquery2}")
    print(f"[DEBUG] Subquery 3: {response.subquery3}")

    queries_to_search = [
        ("original", query),
        ("subquery1", response.subquery1),
        ("subquery2", response.subquery2),
        ("subquery3", response.subquery3)
    ]

    def search_single_query(query_info):
        """Busca documentos para uma única query no Qdrant"""
        query_type, query_text = query_info

        if GEMINI_EMBEDD:
            embedding = normalize(embedding_model.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=[query_text],
                config=genai.types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBED_DIM
                )
            ).embeddings[0].values)
        else:
            embedding = embedding_model.encode(query_text).tolist()

        try:
            results = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=1
            )
            
            # Extrair apenas texto e fonte de cada resultado
            formatted_results = []
            for idx, point in enumerate(results.points, 1):
                texto = point.payload.get('texto', '[Texto não disponível]')
                fonte = point.payload.get('fonte', '[Fonte não disponível]')
                formatted_results.append({
                    'numero': idx,
                    'texto': texto,
                    'fonte': fonte
                })
            
            return {
                "query_type": query_type,
                "query_text": query_text,
                "results": formatted_results
            }
        
        except Exception as e:
            print(f"[ERROR] Erro ao buscar {query_type}: {str(e)}")
            return {
                "query_type": query_type,
                "query_text": query_text,
                "results": []
            }

    # 4 buscas em paralelo 
    with ThreadPoolExecutor(max_workers=4) as executor:
        search_results = list(executor.map(search_single_query, queries_to_search))
    
    # Consolidar resultados de forma limpa e legível
    consolidated_results = []
    for result in search_results:
        query_section = f"\n{'='*80}\n"
        query_section += f"📍 TIPO: {result['query_type'].upper()}\n"
        query_section += f"🔍 CONSULTA: {result['query_text']}\n"
        query_section += f"{'-'*80}\n"
        
        if result['results']:
            for doc in result['results']:
                query_section += f"\n📄 DOCUMENTO {doc['numero']}:\n"
                query_section += f"{doc['texto']}\n\n"
                query_section += f"🔗 FONTE: {doc['fonte']}\n"
                query_section += f"{'-'*80}\n"
        else:
            query_section += "⚠️ Nenhum documento encontrado.\n"
            query_section += f"{'-'*80}\n"
        
        consolidated_results.append(query_section)
    
    final_response = (
        f"\n{'='*80}\n"
        f"📚 DOCUMENTOS RECUPERADOS\n"
        f"{'='*80}\n"
        f"{''.join(consolidated_results)}\n"
        f"{'='*80}\n"
        f"⚠️ IMPORTANTE: Sempre cite a fonte (link) das informações utilizadas.\n"
        f"{'='*80}\n"
    )

    return final_response

#print(retrieve_information.invoke("Quais são as opções de tratamento para sintomas de menopausa e como elas afetam a saúde óssea?"))

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
        return "Não encontrei o email do usuário. Por favor, solicite o email antes de enviar o guia."

    try:
       
        remetente = os.getenv("REMETENTE")
        senha = os.getenv("EMAIL_PASSWORD")
        
        print(f"[DEBUG] Iniciando envio de email para: {email}")
        
        # converter o guide de Markdown para HTML
        guide_html = markdown.markdown(guide, extensions=['extra', 'nl2br'])
        
        # CSS para o PDF
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
        
        print(f"[DEBUG] PDF gerado com sucesso ({len(pdf_buffer.getvalue())} bytes)")
        
        # metadados do email
        msg = MIMEMultipart()
        msg['Subject'] = '🌸 Seu Guia Personalizado Para Consulta'
        msg['From'] = remetente
        msg['To'] = email
        
        # corpo do email
        corpo_email = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #d946a6;">Olá, {nome}! 🌸</h2>
            
            <p>Seu guia personalizado sobre menopausa está pronto!</p>
            
            <p>Preparamos este documento especialmente para você, com base nas informações que você compartilhou. 
            Ele foi criado para ajudá-la a se preparar melhor para sua consulta médica.</p>
            
            <p><strong>📎 O guia está anexado a este email em formato PDF.</strong></p>
            
            <h3 style="color: #e879b9;">💡 Dicas para usar seu guia:</h3>
            <ul>
                <li>Leia o guia com calma antes da consulta</li>
                <li>Faça anotações adicionais se necessário</li>
                <li>Leve-o impresso ou em formato digital para a consulta</li>
                <li>Não hesite em fazer todas as perguntas listadas</li>
            </ul>
            
            <p style="margin-top: 20px;">Desejamos que sua consulta seja produtiva e esclarecedora! 💕</p>
            
            <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
            
            <p style="font-size: 0.9em; color: #666;">
                <em>Este é um email automático. Se você tiver dúvidas ou precisar de ajuda, 
                sinta-se à vontade para conversar comigo novamente!</em>
            </p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(corpo_email, 'html', 'utf-8'))
        
        # anexa pdf
        pdf_attachment = MIMEApplication(pdf_buffer.getvalue(), _subtype='pdf')
        pdf_attachment.add_header(
            'Content-Disposition', 
            'attachment', 
            filename=f'guia_consulta_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
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