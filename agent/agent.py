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

MODEL_NAME = "gemini-3-flash-preview"

def create_agent_graph(checkpointer=None):

    # Configuração do LLM
    llm = ChatGoogleGenerativeAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model=MODEL_NAME,
        temperature=0,
        max_tokens=20000,
        timeout=None,
        max_retries=1,
        transport="rest" 
    )

    graph = StateGraph(state_schema=StateSchema)

    # --- FUNÇÃO AUXILIAR DE NORMALIZAÇÃO ---
    def normalize_content(content):
        """Converte listas de dicionários do Gemini em texto puro string."""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        return str(content)

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
        try:
            response = llm.with_structured_output(RouterOutput).invoke([system_message, *state["messages"]])
            route = response.route
        except Exception:
            route = "chat_node"

        if route not in ["chat_node", "guide_node"]:
            route = "chat_node"
        return {"route": route}

    def chat_node(state: StateSchema) -> StateSchema:
        system_prompt = SystemMessage(content=CHAT_SYSTEM_PROMPT)
        response = llm.bind_tools(tools=TOOLS_CHAT).invoke([system_prompt, *state["messages"]])
        
        # Normaliza a resposta do chat comum
        response.content = normalize_content(response.content)
        
        return {"messages": [response]}

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
                if key == "guide": continue 
                pretty_key = key.replace("_", " ").capitalize()
                val = str(value)
                lines.append(f"• {pretty_key}: {val}\n")
            lines.append(sep)
            lines.append("Se quiser alterar algo, clique em 'Corrigir'. Caso contrário, confirme.")
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
        
        if filtered_data:
            prompt_parts.append("=== DADOS DA PACIENTE ===\n\n")
            for key, value in filtered_data.items():
                label = questions_map.get(key, key)
                val_str = str(value) if value is not None else "Não informado"
                prompt_parts.append(f"{label} {val_str}\n\n")

        prompt_parts.append(
            "\nGere o guia completo seguindo EXATAMENTE o formato especificado no system prompt, "
            "incluindo os marcadores [INICIO_GUIA] e [FIM_GUIA]. "
            "Use as perguntas e respostas acima como contexto para personalizar o guia de forma detalhada e relevante."
        )

        user_message = HumanMessage(content="".join(prompt_parts))

        try:
            response = llm.invoke([system_message, user_message])
            
            # Normaliza o conteúdo AQUI antes de qualquer outra coisa
            content = normalize_content(response.content)
            # Atualiza o objeto response para que o app receba string limpa também
            response.content = content 
            
            guide_content = content
            if "[INICIO_GUIA]" in content and "[FIM_GUIA]" in content:
                try:
                    start_idx = content.find("[INICIO_GUIA]") + len("[INICIO_GUIA]")
                    end_idx = content.find("[FIM_GUIA]")
                    guide_content = content[start_idx:end_idx].strip()
                except:
                    guide_content = content
            
            if "user_data" not in state: state["user_data"] = {}
            # Salva STR, não LISTA.
            state["user_data"]["guide"] = guide_content

            return {
                "messages": [response],
                "user_data": state["user_data"]
            }
        
        except Exception as e:
            error_msg = f"Erro técnico ao gerar guia: {str(e)}"
            print(f"[ERROR AGENT] {error_msg}") 
            return {
                "messages": [AIMessage(content=f"❌ {error_msg}")],
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

    graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools_chat", "__end__": END})
    graph.add_edge("tools_chat", "chat_node")
    graph.add_edge("guide_node", "personal_questions")
    graph.add_edge("personal_questions", "health_questions")
    graph.add_edge("health_questions", "show_user_data_node")
    graph.add_edge("show_user_data_node", "ask_confirmation")

    def data_condition(state: StateSchema) -> Literal["personal_questions", "generate_guide"]:
        return "generate_guide" if state.get("confirmation") else "personal_questions"

    graph.add_conditional_edges("ask_confirmation", data_condition)
    
    def welcome_condition(state: StateSchema) -> Literal["router_node", "welcome_node"]:
        return "welcome_node" if len(state["messages"]) <= 1 else "router_node"

    graph.add_conditional_edges(START, welcome_condition)
    graph.add_edge("generate_guide", END)

    return graph.compile(checkpointer=checkpointer)

graph = create_agent_graph()