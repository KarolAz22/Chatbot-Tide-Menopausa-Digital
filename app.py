import streamlit as st
import os
import dotenv
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver 
from agent.agent import create_agent_graph 

# 1. CARREGAMENTO DE AMBIENTE
dotenv.load_dotenv()
REQUIRED_KEYS = ["GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]

try:
    for key in REQUIRED_KEYS:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

st.set_page_config(page_title="Tide - Menopausa Digital", page_icon="🌸", layout="centered")

missing_keys = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing_keys:
    st.error(f"⚠️ Erro: Chaves faltando: {', '.join(missing_keys)}")
    st.stop()

# --- ESTADO ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "graph" not in st.session_state:
    memory = InMemorySaver()
    st.session_state.graph = create_agent_graph(checkpointer=memory)

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("🌸 Tide: Seu Guia Digital da Menopausa")

# --- HISTÓRICO ---
for message in st.session_state.messages:
    if message.get("role") == "tool_log":
        with st.status(message["content"], state="complete"):
            st.write("Consulta realizada.")
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- EXECUÇÃO ---
def run_graph(input_data):
    with st.chat_message("assistant"):
        status_container = st.status("Processando...", expanded=True)
        response_text = ""
        
        try:
            for event in st.session_state.graph.stream(input_data, config, stream_mode="values"):
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if tool_call["name"] == "retrieve_information":
                                query = tool_call['args'].get('query', 'consulta')
                                status_container.write(f"🔍 Pesquisando: *{query}*")
                            elif tool_call["name"] == "send_pdf":
                                status_container.write("📧 Preparando e enviando e-mail...")

                    if isinstance(last_message, ToolMessage):
                        status_container.write("✅ Dados recebidos.")

                    if isinstance(last_message, AIMessage) and last_message.content:
                        if not last_message.tool_calls:
                            raw_content = last_message.content
                            
                            # PROTEÇÃO: Garante que dicionários virem texto antes de exibir
                            if isinstance(raw_content, list):
                                text_parts = []
                                for item in raw_content:
                                    if isinstance(item, dict) and 'text' in item:
                                        text_parts.append(item['text'])
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                raw_content = "".join(text_parts)
                            
                            raw_content = str(raw_content)
                            cleaned_content = raw_content.replace("```markdown", "").replace("```", "")
                            cleaned_content = cleaned_content.replace("[INICIO_GUIA]", "").replace("[FIM_GUIA]", "")
                            response_text = cleaned_content.strip()
            
            status_container.update(label="Respondido!", state="complete", expanded=False)
            
            if response_text:
                st.markdown(response_text)
                if not st.session_state.messages or st.session_state.messages[-1].get("content") != response_text:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
        except Exception as e:
            status_container.update(label="Erro na execução", state="error")
            st.error(f"Detalhe do erro: {e}")

# --- INTERFACE DINÂMICA ---
try:
    state_snapshot = st.session_state.graph.get_state(config)
    
    if state_snapshot.next:
        current_node = state_snapshot.next[0] if isinstance(state_snapshot.next, tuple) else state_snapshot.next
        
        # === FORMULÁRIOS ===
        if current_node == "personal_questions":
            with st.chat_message("assistant"):
                st.write("📝 **Preciso de alguns dados para continuar:**")
                with st.form("form_pessoal"):
                    nome = st.text_input("Qual é seu nome?")
                    idade = st.text_input("Qual é sua idade?") 
                    email = st.text_input("Qual é o seu email?")
                    if st.form_submit_button("Enviar Dados"):
                        if not nome or not idade or not email:
                            st.warning("⚠️ Preencha todos os campos.")
                        else:
                            run_graph(Command(resume={"nome": nome, "idade": str(idade), "email": email}))
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
                        if not all([c1, c2, c3, c4]):
                            st.warning("⚠️ Preencha os campos.")
                        else:
                            run_graph(Command(resume={
                                "ciclo_menstrual": c1, "sintomas_fisicos": c2,
                                "saude_emocional": c3, "habitos_historico": c4,
                                "exames_tratamentos": c5
                            }))
                            st.rerun()

        elif current_node == "ask_confirmation":
             with st.chat_message("assistant"):
                st.info("As informações acima estão corretas?")
                col1, col2 = st.columns(2)
                if col1.button("✅ Sim, Gerar Guia"):
                    run_graph(Command(resume={"confirmation": True}))
                    st.rerun()
                if col2.button("❌ Corrigir"):
                    run_graph(Command(resume={"confirmation": False}))
                    st.rerun()
    else:
        if prompt := st.chat_input("Tire suas dúvidas sobre menopausa..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            run_graph({"messages": [HumanMessage(content=prompt)]})

except Exception:
    if prompt := st.chat_input("Diga 'Olá' para começar"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        run_graph({"messages": [HumanMessage(content=prompt)]})