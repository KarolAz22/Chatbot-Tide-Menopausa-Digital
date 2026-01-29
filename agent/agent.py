import os
import json
from typing import Literal

from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from agent.utils.prompt import CHAT_SYSTEM_PROMPT, WELCOME_MESSAGE, ROUTER_PROMPT, GUIDE_SYSTEM_PROMPT
from agent.utils.state import StateSchema
from agent.utils.tools import TOOLS_CHAT

MODEL_NAME = "gemini-2.5-flash-lite"

def create_agent_graph(checkpointer=None): #Todo

    llm = ChatGoogleGenerativeAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model=MODEL_NAME,
        temperature=0,
        max_tokens=20000,
        timeout=None,
        max_retries=1,            
    )

    graph = StateGraph(state_schema=StateSchema)

    # --- NODES ---

    def welcome_node(state: StateSchema) -> StateSchema:

        state["confirmation"] = False

        return {
            "messages": [AIMessage(content=WELCOME_MESSAGE)]
        }
  

    def router_node(state: StateSchema) -> str:

        class RouterOutput(BaseModel):
            route: str

        system_message = SystemMessage(content=ROUTER_PROMPT)
        
        # Otimização: A router decide o fluxo macro
        response = llm.with_structured_output(RouterOutput).invoke([system_message, *state["messages"]])

        route = response.route
        if route not in ["chat_node", "guide_node"]:
            route = "chat_node"
            
        return {"route": route}

    def chat_node(state: StateSchema) -> StateSchema:

        system_prompt = SystemMessage(content=CHAT_SYSTEM_PROMPT)
        
        # Bind de ferramentas
        response = llm.bind_tools(tools=TOOLS_CHAT).invoke([system_prompt, *state["messages"]])

        # Normalizar o conteúdo da resposta se vier fragmentado (comum em streaming/tools)
        if hasattr(response, 'content') and isinstance(response.content, list):
            text_parts = []
            for item in response.content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            response.content = ''.join(text_parts)

        return {
            "messages": [response],
        }

    # --- NÓS DO FLUXO DE GUIA (Mantidos conforme original) ---
    def guide_node(state: StateSchema) -> StateSchema:

        return {
            "messages": [AIMessage(content="Antes de prosseguirmos, gostaria de fazer algumas perguntas para personalizar melhor o guia para você.")]
        }
    
    def personal_questions(state: StateSchema) -> StateSchema:

        user_data = state.get("user_data", {})

        questions_prompt = (
            "Por favor, responda as seguintes perguntas pessoais:\n\n"
            "1. Qual é seu nome?\n"
            "2. Qual é sua idade?\n"
            "3. Qual é o seu email? (Usaremos para enviar o guia personalizado)\n\n"
        )
        answer = interrupt(questions_prompt)

        user_data["nome"] = answer.get("nome", "Não informado")
        user_data["idade"] = answer.get("idade", "Não informado")
        user_data["email"] = answer.get("email", "Não informado")
        return {"user_data": user_data}

    def health_questions(state: StateSchema) -> StateSchema:

        user_data = state.get("user_data", {})

        questions_prompt = (
            "Agora, por favor responda as seguintes perguntas sobre sua saúde:\n\n"
            "1. Como está o seu ciclo menstrual? Ela tem sido regular em frequência e fluxo? "
            "Você já completou 12 meses consecutivos sem menstruar?\n\n"
            "2. Quais sintomas físicos novos ou incômodos você tem sentido? "
            "(Por exemplo: ondas de calor, suores noturnos, alterações no sono, cansaço, ressecamento vaginal, "
            "mudanças na libido, ganho de peso, queda de cabelo ou infecções urinárias)\n\n"
            "3. Como você tem se sentido emocional e mentalmente? "
            "(Flutuações de humor, ansiedade, irritabilidade, desânimo, dificuldade de memória e concentração)\n\n"
            "4. Como estão seus hábitos de saúde e histórico médico? "
            "(Medicamentos ou suplementos que você usa, histórico pessoal ou familiar de doenças crônicas, "
            "especialmente câncer de mama, rotina de alimentação, exercícios, consumo de álcool ou fumo)\n\n"
            "5. Quando você realizou seus últimos exames preventivos e quais tratamentos você gostaria de discutir? "
            "(Papanicolau, mamografia e densitometria óssea. Você já tentou algo para os sintomas ou tem interesse "
            "em discutir opções, como a terapia de reposição hormonal?)\n\n"
        )

        answer = interrupt(questions_prompt)

        user_data["ciclo_menstrual"] = answer.get("ciclo_menstrual", "Não informado")
        user_data["sintomas_fisicos"] = answer.get("sintomas_fisicos", "Não informado")
        user_data["saude_emocional"] = answer.get("saude_emocional", "Não informado")
        user_data["habitos_historico"] = answer.get("habitos_historico", "Não informado")
        user_data["exames_tratamentos"] = answer.get("exames_tratamentos", "Não informado")
        return {"user_data": user_data}

    def show_user_data_node(state: StateSchema) -> StateSchema:
        user_data = state.get("user_data", {}) or {}

        if not user_data:
            content = (
                "Ainda não recebi informações suas. Quando estiver pronto, posso fazer as perguntas novamente."
            )
        else:
            header = "Obrigado por fornecer essas informações. Aqui está um resumo dos dados que você compartilhou:\n"
            sep = "────────────────────────────────────────\n"

            lines = [header, sep]

            for key, value in user_data.items():
                # torna a chave mais legível: 'tempo_menopausa' -> 'Tempo menopausa'
                pretty_key = key.replace("_", " ").capitalize()
                val = ", ".join(f"{k}: {v}" for k, v in value.items()) if isinstance(value, dict) else str(value)
                lines.append(f"• {pretty_key}: {val}\n")

            lines.append(sep)
            lines.append("Se quiser alterar algum item, clique em ignorar para recomeçar.")

            content = "\n".join(lines)

        return {"messages": [AIMessage(content=content)]}

        

    def ask_confirmation(state: StateSchema) -> StateSchema:

        question = "Voce confirma que essas informações estão corretas e completas para prosseguirmos com o guia?"

        answer = interrupt(question)
        return {"confirmation": answer["confirmation"]}

    def generate_guide(state: StateSchema) -> StateSchema:

        user_data = state.get("user_data", {}) or {}

        system_message = SystemMessage(content=GUIDE_SYSTEM_PROMPT)

        # Mapeamento das perguntas feitas ao usuário
        questions_map = {
            "email": "Qual é o seu email? (Usaremos para enviar o guia personalizado)",
            "nome": "Qual é seu nome?",
            "idade": "Qual é sua idade?",
            "ciclo_menstrual": "Como está o seu ciclo menstrual? (Quando foi sua última menstruação, ela tem sido regular em frequência e fluxo? Você já completou 12 meses consecutivos sem menstruar?)",
            "sintomas_fisicos": "Quais sintomas físicos novos ou incômodos você tem sentido? (Por exemplo: ondas de calor, suores noturnos, alterações no sono, cansaço, ressecamento vaginal, mudanças na libido, ganho de peso, queda de cabelo ou infecções urinárias?)",
            "saude_emocional": "Como você tem se sentido emocional e mentalmente? (Você notou flutuações de humor, ansiedade, irritabilidade, desânimo, ou dificuldade de memória e concentração?)",
            "habitos_historico": "Como estão seus hábitos de saúde e histórico médico? (Incluindo medicamentos ou suplementos que você usa, seu histórico pessoal ou familiar de doenças crônicas, especialmente câncer de mama, sua rotina de alimentação, exercícios, consumo de álcool ou fumo.)",
            "exames_tratamentos": "Quando você realizou seus últimos exames preventivos e quais tratamentos você gostaria de discutir? (Como Papanicolau, mamografia e densitometria óssea. Você já tentou algo para os sintomas ou tem interesse em discutir opções, como a terapia de reposição hormonal?)"
        }

        prompt_parts = [
            "Crie um guia personalizado de menopausa com base nas seguintes informações coletadas:\n\n"
        ]

        filtered_data = {k: v for k, v in user_data.items() if k != "guide"}

        if not filtered_data or len(filtered_data) == 0:
            # se não houver dados, criar um guia genérico
            prompt_parts.append("Informações do paciente: Dados não informados\n")
        else:
            prompt_parts.append("=== PERGUNTAS E RESPOSTAS DA PACIENTE ===\n\n")
            for key, value in filtered_data.items():
                # Adiciona a pergunta correspondente
                question = questions_map.get(key, key.replace("_", " ").capitalize())
                
                if value and value != "Não informado":
                    prompt_parts.append(f"PERGUNTA: {question}\n")
                    prompt_parts.append(f"RESPOSTA: {value}\n\n")

        prompt_parts.append(
            "\nGere o guia completo seguindo EXATAMENTE o formato especificado no system prompt, "
            "incluindo os marcadores [INICIO_GUIA] e [FIM_GUIA]. "
            "Use as perguntas e respostas acima como contexto para personalizar o guia de forma detalhada e relevante."
        )

        user_message = HumanMessage(content="".join(prompt_parts))

        try:
            response = llm.invoke([system_message, user_message])
            
            if not response or not response.content:
                # Fallback se não houver conteúdo
                fallback_guide_content = (
                    "# Guia Personalizado para Consulta sobre Menopausa\n\n"
                    "## 📋 Informações da Paciente\n"
                    "Informações não fornecidas.\n\n"
                    "## 🔍 Resumo da Situação Atual\n"
                    "Este guia foi criado para ajudá-la a preparar sua consulta médica sobre menopausa.\n\n"
                    "## 🩺 Sintomas e Observações\n"
                    "- Sintomas não especificados\n\n"
                    "## ❓ Perguntas Importantes para o Médico\n"
                    "1. Quais são os sintomas mais comuns da menopausa?\n"
                    "2. Quais tratamentos estão disponíveis para mim?\n"
                    "3. Como posso melhorar minha qualidade de vida durante este período?\n"
                    "4. Existem mudanças no estilo de vida que você recomenda?\n"
                    "5. Quando devo retornar para acompanhamento?\n\n"
                    "## 💡 Recomendações de Bem-Estar\n"
                    "- Mantenha uma alimentação equilibrada rica em cálcio e vitamina D\n"
                    "- Pratique exercícios físicos regularmente\n"
                    "- Cuide da saúde mental e busque apoio quando necessário\n"
                    "- Mantenha-se hidratada\n\n"
                    "## 📌 Próximos Passos\n"
                    "- Anote qualquer sintoma novo antes da consulta\n"
                    "- Leve este guia impresso ou em formato digital\n"
                    "- Não hesite em fazer todas as suas perguntas ao médico\n\n"
                    "---\n"
                    "*Este guia foi gerado para auxiliar na preparação da sua consulta médica.*"
                )
                
                full_response = (
                    f"[INICIO_GUIA]\n{fallback_guide_content}\n[FIM_GUIA]\n\n"
                    "Pronto! Seu guia personalizado foi gerado com sucesso! 📋✨ "
                    "Gostaria que eu enviasse este guia para o seu email?"
                )
                
                response = AIMessage(content=full_response)
            
            content = response.content
            guide_content = content
            
            if "[INICIO_GUIA]" in content and "[FIM_GUIA]" in content:
                start_idx = content.find("[INICIO_GUIA]") + len("[INICIO_GUIA]")
                end_idx = content.find("[FIM_GUIA]")
                guide_content = content[start_idx:end_idx].strip()
            
            if "user_data" not in state:
                state["user_data"] = {}
            state["user_data"]["guide"] = guide_content

            return {
                "messages": [response],
                "user_data": state["user_data"]
            }
        
        except Exception as e:
            #print(f"[ERROR] Erro ao gerar guia: {str(e)}")
           
            error_message = AIMessage(
                content=f"Desculpe, houve um problema ao gerar o guia. Por favor, tente novamente mais tarde. Se o problema persistir, entre em contato com o suporte."
            )
            return {
                "messages": [error_message],
                "user_data": state.get("user_data", {})
            }

    tool_node = ToolNode(tools=TOOLS_CHAT, name="tools_chat")
    
    graph.add_node("welcome_node", welcome_node)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools_chat", tool_node)
    graph.add_node("router_node", router_node)
    graph.add_node("guide_node", guide_node)
    graph.add_node("personal_questions", personal_questions)
    graph.add_node("health_questions", health_questions)
    graph.add_node("show_user_data_node", show_user_data_node)
    graph.add_node("ask_confirmation", ask_confirmation)
    graph.add_node("generate_guide", generate_guide)



   
    # Definição de arestas
    graph.add_edge("welcome_node", END)

    # Fluxo Router
    def route_condition(state: StateSchema) -> Literal["chat_node", "guide_node"]:
        if state.get("route") == "chat_node":
            return "chat_node"
        return "guide_node"

    graph.add_conditional_edges("router_node", route_condition)

    # Fluxo Chat (OTIMIZADO)
    # Aqui removemos a lógica de avaliação. Se usar tool, vai pra tool. Se não, encerra a rodada.
    graph.add_conditional_edges(
        "chat_node", 
        tools_condition, 
        {"tools": "tools_chat", "__end__": END}
    )
    graph.add_edge("tools_chat", "chat_node")

    # Fluxo Guia (Linear)
    graph.add_edge("guide_node", "personal_questions")
    graph.add_edge("personal_questions", "health_questions")
    graph.add_edge("health_questions", "show_user_data_node")
    graph.add_edge("show_user_data_node", "ask_confirmation")



    def data_condition(state: StateSchema) -> Literal["personal_questions", "generate_guide"]:
        if state.get("confirmation"):
            return "generate_guide"
        return "personal_questions"

    graph.add_conditional_edges("ask_confirmation", data_condition)
    def welcome_condition(state:  StateSchema) -> Literal["router_node", "welcome_node"]:

        if len(state["messages"]) <= 1:
            return "welcome_node"
        else:
            return "router_node"

    graph.add_conditional_edges(START, welcome_condition)

    graph.add_edge("generate_guide", END)

    return graph.compile(checkpointer=checkpointer) #Todo

graph = create_agent_graph()