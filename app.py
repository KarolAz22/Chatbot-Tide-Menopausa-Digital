import streamlit as st
import os
import dotenv
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver # Necessário para memória
from agent.agent import create_agent_graph # Importamos a função criadora, não o grafo pronto

# 1. CARREGAMENTO DE VARIÁVEIS DE AMBIENTE
dotenv.load_dotenv()

REQUIRED_KEYS = ["GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]

for key in REQUIRED_KEYS:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# Configuração da Página
st.set_page_config(page_title="Tide - Menopausa Digital", page_icon="🤖", layout="centered")

# --- DEBUG DE AMBIENTE ---
missing_keys = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing_keys:
    st.error(f"⚠️ Erro de Configuração: Chaves faltando: {', '.join(missing_keys)}")
    st.stop()

# --- INICIALIZAÇÃO DO ESTADO ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

# --- INICIALIZAÇÃO DO GRAFO (CORREÇÃO DE MEMÓRIA) ---
if "graph" not in st.session_state:
    # Cria uma memória exclusiva para esta sessão
    memory = InMemorySaver()
    # Cria o grafo injetando essa memória
    st.session_state.graph = create_agent_graph(checkpointer=memory)

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("🤖 Tide: Seu Guia Digital da Menopausa")

# --- EXIBIÇÃO DO HISTÓRICO ---
for message in st.session_state.messages:
    if message.get("role") == "tool_log":
        with st.status(message["content"], state="complete"):
            st.write("Consulta realizada.")
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- FUNÇÃO PRINCIPAL DE EXECUÇÃO ---
def run_graph(input_data):
    """Executa o grafo (seja nova mensagem ou resume de interrupt)"""
    
    with st.chat_message("assistant"):
        status_container = st.status("Processando...", expanded=True)
        response_text = ""
        
        try:
            # Usa o grafo armazenado na sessão
            # stream_mode="values" retorna a lista completa de mensagens atualizada
            for event in st.session_state.graph.stream(input_data, config, stream_mode="values"):
                
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    
                    # 1. DETECTA USO DE FERRAMENTA
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if tool_call["name"] == "retrieve_information":
                                query = tool_call['args'].get('query', 'consulta')
                                status_container.write(f"🔍 Pesquisando: *{query}*")
                            elif tool_call["name"] == "send_pdf":
                                status_container.write("📧 Enviando email...")

                    # 2. DETECTA RESPOSTA DA FERRAMENTA
                    if isinstance(last_message, ToolMessage):
                        status_container.write("✅ Dados recebidos.")

                    # 3. DETECTA RESPOSTA FINAL (AI)
                    if isinstance(last_message, AIMessage) and last_message.content:
                        if not last_message.tool_calls:
                            response_text = last_message.content
            
            status_container.update(label="Respondido!", state="complete", expanded=False)
            
            if response_text:
                st.markdown(response_text)
                # Verifica se a mensagem já não está no histórico para evitar duplicação
                if not st.session_state.messages or st.session_state.messages[-1]["content"] != response_text:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
        except Exception as e:
            status_container.update(label="Aguardando ação...", state="complete", expanded=False)
            # Erros de interrupção são normais no LangGraph, não precisamos mostrar erro vermelho
            # print(f"Interrupção ou erro: {e}") 

# --- LÓGICA DE INTERFACE DINÂMICA (INTERRUPTS) ---
# Verifica o estado atual do grafo para ver se parou num interrupt
try:
    state_snapshot = st.session_state.graph.get_state(config)
    
    if state_snapshot.next:
        # Se houver próximo passo e estiver parado, é um interrupt
        current_node = state_snapshot.next[0] if isinstance(state_snapshot.next, tuple) else state_snapshot.next
        
        # === FORMULÁRIOS ===
        if current_node == "personal_questions":
            with st.chat_message("assistant"):
                st.write("📝 **Preciso de alguns dados para continuar:**")
                with st.form("form_pessoal"):
                    nome = st.text_input("Qual é seu nome?")
                    idade = st.text_input("Qual é sua idade?") # Melhor ser text para evitar erro de tipo no json
                    email = st.text_input("Qual é o seu email?")
                    
                    if st.form_submit_button("Enviar Dados"):
                        # CORREÇÃO CRÍTICA: Usamos Command(resume=...) para destravar o interrupt
                        dados = {"nome": nome, "idade": str(idade), "email": email}
                        run_graph(Command(resume=dados))
                        st.rerun()

        elif current_node == "health_questions":
             with st.chat_message("assistant"):
                st.write("🩺 **Sobre sua saúde:**")
                with st.form("form_saude"):
                    c1 = st.text_area("Ciclo Menstrual", placeholder="Frequência, fluxo...")
                    c2 = st.text_area("Sintomas Físicos", placeholder="Calorões, insônia...")
                    c3 = st.text_area("Saúde Emocional", placeholder="Ansiedade, humor...")
                    c4 = st.text_area("Histórico e Hábitos", placeholder="Medicamentos, histórico familiar...")
                    c5 = st.text_area("Exames e Tratamentos", placeholder="Últimos exames...")
                    
                    if st.form_submit_button("Gerar Guia"):
                        dados_saude = {
                            "ciclo_menstrual": c1, "sintomas_fisicos": c2,
                            "saude_emocional": c3, "habitos_historico": c4,
                            "exames_tratamentos": c5
                        }
                        run_graph(Command(resume=dados_saude))
                        st.rerun()

        elif current_node == "ask_confirmation":
             with st.chat_message("assistant"):
                st.info("Confirma que os dados estão corretos?")
                col1, col2 = st.columns(2)
                if col1.button("✅ Confirmar"):
                    run_graph(Command(resume={"confirmation": True}))
                    st.rerun()
                if col2.button("❌ Corrigir"):
                    run_graph(Command(resume={"confirmation": False}))
                    st.rerun()

    # --- CHAT INPUT (Só aparece se NÃO estiver num formulário) ---
    else:
        if prompt := st.chat_input("Tire suas dúvidas sobre menopausa..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            run_graph({"messages": [HumanMessage(content=prompt)]})

except Exception as e:
    # Caso inicial onde o grafo ainda não rodou nada
    if prompt := st.chat_input("Digite 'olá' para começar..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        run_graph({"messages": [HumanMessage(content=prompt)]})