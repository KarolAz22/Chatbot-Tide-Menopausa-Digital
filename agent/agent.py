from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic import BaseModel
from typing import Literal
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from agent.utils.prompt import CHAT_SYSTEM_PROMPT, WELCOME_MESSAGE, ROUTER_PROMPT, GUIDE_SYSTEM_PROMPT
from agent.utils.state import StateSchema
from agent.utils.tools import TOOLS_CHAT
import json

MODEL_NAME = "gemini-2.5-flash"

def create_agent_graph():

    llm =  ChatGoogleGenerativeAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model=MODEL_NAME,
        temperature=0,
        max_tokens=20000,
        timeout=None,
        max_retries=2,            
    )

    graph = StateGraph(state_schema=StateSchema)

    def welcome_node(state: StateSchema) -> StateSchema:

        state["confirmation"] = False

        return {
            "messages": [AIMessage(content=WELCOME_MESSAGE)]
        }
  

    def router_node(state: StateSchema) -> str:

        class RouterOutput(BaseModel):
            route: str

        system_message = SystemMessage(content=ROUTER_PROMPT)

        response = llm.with_structured_output(RouterOutput).invoke([system_message, *state["messages"]])

        state["route"] = response.route

        if state["route"] not in ["chat_node", "guide_node"]:
            state["route"] = "chat_node"

        return state
    

    def chat_node(state: StateSchema) -> StateSchema:

        system_prompt = SystemMessage(content=CHAT_SYSTEM_PROMPT)

        response =  llm.bind_tools(tools=TOOLS_CHAT).invoke([system_prompt, *state["messages"]])

        # Normalizar o conteúdo da resposta se vier como array
        if hasattr(response, 'content') and isinstance(response.content, list):
            # Concatenar todos os textos do array de conteúdo
            text_parts = []
            for item in response.content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            
            # Criar nova mensagem com conteúdo unificado
            response.content = ''.join(text_parts)

        return {
            "messages": [response],
        }
    
    
    def evaluate_response_node(state: StateSchema) -> StateSchema:
        """Avalia se a resposta do chat está adequada"""
        
        class EvaluationOutput(BaseModel):
            pass_evaluation: bool
            problem: str
        
        evaluation_prompt = """Você é um avaliador de qualidade de respostas sobre menopausa e saúde feminina.

        ⚠️ IMPORTANTE: AVALIE APENAS A ÚLTIMA MENSAGEM DO ASSISTENTE (a mais recente na conversa).
        IGNORE completamente mensagens anteriores do assistente - elas são apenas contexto histórico.

        O histórico da conversa está disponível apenas para você entender o contexto, mas você deve avaliar EXCLUSIVAMENTE a resposta mais recente do assistente.

        Critérios de avaliação para A ÚLTIMA MENSAGEM DO ASSISTENTE:
        1. A resposta está clara, educada e bem estruturada?
        2. As informações são precisas e relevantes ao contexto da pergunta atual?
        3. Se ferramentas foram chamadas, os resultados foram bem utilizados?
        4. A resposta atende adequadamente à pergunta do usuário?
        5. A resposta NÃO está vazia ou incompleta?

        Retorne:
        - pass_evaluation: true se a ÚLTIMA resposta está adequada, false se precisa de melhorias graves ou reformulação
        - problem: string vazia se pass_evaluation=true, ou uma descrição específica e objetiva do que precisa ser melhorado NA ÚLTIMA RESPOSTA

        Não seja excessivamente rigoroso com detalhes menores. Foque em problemas críticos que realmente comprometem a qualidade da resposta."""

        system_message = SystemMessage(content=evaluation_prompt)
        
        response = llm.with_structured_output(EvaluationOutput).invoke([system_message, *state["messages"]])
        
        return {
            "pass_evaluation": response.pass_evaluation,
            "problem": response.problem
        }
    
    
    def reformulate_response_node(state: StateSchema) -> StateSchema:
        """Reformula a resposta com base no problema identificado"""
        
        reformulation_prompt = f"""Você é um assistente especializado em saúde feminina e menopausa.

        ⚠️ ATENÇÃO: Você DEVE reformular APENAS A ÚLTIMA MENSAGEM que você (o assistente) enviou.

        PROBLEMA IDENTIFICADO NA ÚLTIMA RESPOSTA:
        {state.get("problem", "Resposta precisa ser melhorada")}

        📋 INSTRUÇÕES:
        1. Analise o histórico da conversa para entender o contexto
        2. Identifique qual foi a ÚLTIMA pergunta/solicitação do usuário
        3. Reformule APENAS sua última resposta para corrigir o problema identificado
        4. NÃO repita ou reformule respostas antigas - foque exclusivamente na mais recente
        5. Sua nova resposta NÃO pode estar vazia

        ✅ Mantenha na resposta reformulada:
        - Relevância ao contexto atual da conversa
        - Informações precisas e empáticas
        - Tom adequado ao tema de saúde feminina
        - Clareza e completude
        - Educação e profissionalismo

        Retorne APENAS a resposta reformulada, sem explicações adicionais sobre o que você mudou."""

        system_message = SystemMessage(content=reformulation_prompt)
        
        response = llm.invoke([system_message, *state["messages"], HumanMessage(content="Reformule sua última resposta agora corrigindo o problema identificado. Retorne APENAS a resposta reformulada.")])        
        # Remove a última mensagem (resposta inadequada) e adiciona a reformulada
        new_messages = state["messages"][:-1] + [response]
        
        return {
            "messages": new_messages,
            "pass_evaluation": False,  # Reset para nova avaliação
            "problem": ""  # Limpa o problema
        }
    
    
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

        # return a dictionary structured
        answer = interrupt(questions_prompt)

        user_data["nome"] = answer.get("nome", "Não informado")
        user_data["idade"] = answer.get("idade", "Não informado")
        user_data["email"] = answer.get("email", "Não informado")

        state["user_data"] = user_data

        #print(f"[DEBUG] Dados pessoais coletados: {user_data}")

        return state
    
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

        state["user_data"] = user_data

       # print(f"[DEBUG] Dados de saúde coletados: {user_data}")

        return state

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

                # formata valores compostos (por exemplo, dicts) de forma compacta
                if isinstance(value, dict):
                    val = ", ".join(f"{k}: {v}" for k, v in value.items())
                else:
                    val = str(value)

                lines.append(f"• {pretty_key}: {val}\n")

            lines.append(sep)
            lines.append("Se quiser alterar algum item, clique em ignorar para recomeçar.")

            content = "\n".join(lines)

        return {"messages": [AIMessage(content=content)]}

        

    def ask_confirmation(state: StateSchema) -> StateSchema:

        question = "Voce confirma que essas informações estão corretas e completas para prosseguirmos com o guia?"

        answer = interrupt(question)

        state["confirmation"] = answer["confirmation"]

        return state
    
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
            #print(f"[DEBUG] Gerando guia com dados: {filtered_data}")
            
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
            
            #print(f"[DEBUG] Guia salvo com {len(guide_content)} caracteres")

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
    
    graph.add_node(welcome_node, name="welcome_node")
    graph.add_node(chat_node, name="chat_node")
    graph.add_node(tool_node, name="tools_chat")
    graph.add_node(router_node, name="router_node")
    graph.add_node(guide_node, name="guide_node")
    graph.add_node(personal_questions, name="personal_questions")
    graph.add_node(health_questions, name="health_questions")
    graph.add_node(show_user_data_node, name="show_user_data_node")
    graph.add_node(ask_confirmation, name="ask_confirmation")
    graph.add_node(generate_guide, name="generate_guide")
    graph.add_node(evaluate_response_node, name="evaluate_response_node")
    graph.add_node(reformulate_response_node, name="reformulate_response_node")



   
 
    graph.add_edge("welcome_node", END)
    graph.add_edge("guide_node", "personal_questions")
    graph.add_edge("personal_questions", "health_questions")
    graph.add_edge("health_questions", "show_user_data_node")
    graph.add_edge("show_user_data_node", "ask_confirmation")
    graph.add_edge("tools_chat", "chat_node")
    graph.add_edge("generate_guide", END)
    graph.add_edge("reformulate_response_node", "evaluate_response_node")

    graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools_chat", "__end__": "evaluate_response_node"})


    def data_condition(state:  StateSchema) -> Literal["personal_questions", "generate_guide"]:

        if state.get("confirmation"):
            return "generate_guide"
        else:
            return "personal_questions"

    graph.add_conditional_edges("ask_confirmation", data_condition)
    
    def evaluation_condition(state: StateSchema) -> Literal["reformulate_response_node", "__end__"]:
        """Decide se a resposta precisa ser reformulada ou está aprovada"""
        
        if state.get("pass_evaluation", False):
            return "__end__"
        else:
            return "reformulate_response_node"
    
    graph.add_conditional_edges("evaluate_response_node", evaluation_condition)

    def welcome_condition(state:  StateSchema) -> Literal["router_node", "welcome_node"]:

        if len(state["messages"]) == 1:
            return "welcome_node"
        else:
            return "router_node"

    graph.add_conditional_edges(START, welcome_condition)

    def route_condition(state:  StateSchema) -> Literal["chat_node", "guide_node"]:

        if state.get("route") == "chat_node":
            return "chat_node"
        else:
            return "guide_node"
        
    graph.add_conditional_edges("router_node", route_condition)


    compiled_graph = graph.compile()


    return compiled_graph


graph = create_agent_graph()
