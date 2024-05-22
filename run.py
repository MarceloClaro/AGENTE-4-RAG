import json
import streamlit as st
import os
from typing import Tuple
from groq import Groq

st.set_page_config(layout="wide")

FILEPATH = "agents.json"
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192, 
    'llama3-8b-8192': 8192,
    'llama2-70b-4096': 4096,
    'gemma-7b-it': 8192,
}

def load_agent_options() -> list:
    agent_options = ['Criar (ou escolher) um especialista...']
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de agentes. Por favor, verifique o formato.")
    return agent_options

def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

def refresh_page():
    st.rerun()

def save_expert(expert_title: str, expert_description: str):
    with open(FILEPATH, 'r+') as file:
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)
        json.dump(agents, file, indent=4)
        file.truncate()

def fetch_assistant_response(user_input: str, model_name: str, temperature: float, agent_selection: str, groq_api_key: str) -> Tuple[str, str]:
    phase_two_response = ""
    expert_title = ""

    try:
        client = Groq(api_key=groq_api_key)

        def get_completion(prompt: str) -> str:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=get_max_tokens(model_name),
                top_p=1,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content

        if agent_selection == "Criar (ou escolher) um especialista...":
            phase_one_prompt = f"Atue como engenheiro de prompt especialista. Analise a seguinte entrada para determinar o título e as características do melhor especialista para responder à pergunta. Comece a resposta com o título do especialista seguido de um ponto ['.'], depois forneça uma descrição concisa desse especialista: {user_input}"
            phase_one_response = get_completion(phase_one_prompt)
            first_period_index = phase_one_response.find(".")
            expert_title = phase_one_response[:first_period_index].strip()
            expert_description = phase_one_response[first_period_index + 1:].strip()
            save_expert(expert_title, expert_description)
        else:
            with open(FILEPATH, 'r') as file:
                agents = json.load(file)
                agent_found = next((agent for agent in agents if agent["agente"] == agent_selection), None)
                if agent_found:
                    expert_title = agent_found["agente"]
                    expert_description = agent_found["descricao"]
                else:
                    raise ValueError("Especialista selecionado não encontrado no arquivo.")

        phase_two_prompt = f"Atue como {expert_title}, um especialista no assunto, e forneça uma resposta completa e bem formatada para a seguinte pergunta: {user_input}"
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

def refine_response(expert_title: str, phase_two_response: str, user_input: str, model_name: str, temperature: float, groq_api_key: str, references_file):
    try:
        client = Groq(api_key=groq_api_key)

        def get_completion(prompt: str) -> str:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=get_max_tokens(model_name),
                top_p=1,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content

        refine_prompt = f"Atue como {expert_title}, um especialista no assunto. Aqui está a resposta original à pergunta '{user_input}': {phase_two_response}\n\nPor favor, revise e refine completamente esta resposta, fazendo melhorias e abordando quaisquer deficiências. Retorne uma versão atualizada da resposta que incorpore seus refinamentos."

        # Adiciona um prompt mais detalhado se não houver referências
        if not references_file:
            refine_prompt += "\n\nComo não há um arquivo de referências fornecido, certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas."

        refined_response = get_completion(refine_prompt)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

def evaluate_response_with_rag(user_input: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, groq_api_key: str) -> str:
    try:
        client = Groq(api_key=groq_api_key)

        def get_completion(prompt: str) -> str:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=get_max_tokens(model_name),
                top_p=1,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content

        rag_prompt = f"Atue como o Rational Agent Generator (RAG) e avalie a resposta do especialista. Aqui está a descrição do especialista:\n{expert_description}\n\nAqui está a pergunta original:\n{user_input}\n\nE aqui está a resposta do especialista:\n{assistant_response}\n\nPor favor, forneça uma avaliação completa da qualidade e precisão da resposta, levando em consideração a descrição do especialista e a resposta fornecida."

        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

agent_options = load_agent_options()

st.title("Agentes Experts IV com Avaliação RAG de resposta")
st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", "", key="entrada_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    groq_api_key = st.text_input("Chave da API Groq:", key="groq_api_key")
    max_tokens = get_max_tokens(model_name)
    st.write(f"Número Máximo de Tokens para o modelo selecionado: {max_tokens}")

    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    evaluate_clicked = st.button("Avaliar Resposta com RAG")
    refresh_clicked = st.button("Atualizar")

    references_file = st.file_uploader("Upload do arquivo JSON com referências (opcional)", type="json", key="arquivo_referencias")

with col2:
    if 'resposta_assistente' not in st.session_state:
        st.session_state.resposta_assistente = ""
    if 'descricao_especialista_ideal' not in st.session_state:
        st.session_state.descricao_especialista_ideal = ""
    if 'resposta_refinada' not in st.session_state:
        st.session_state.resposta_refinada = ""
    if 'resposta_original' not in st.session_state:
        st.session_state.resposta_original = ""
    if 'rag_resposta' not in st.session_state:
        st.session_state.rag_resposta = ""

    container_saida = st.container()

    if fetch_clicked:
        if references_file is None:
            st.warning("Não foi fornecido um arquivo de referências. Certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas.")
        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, model_name, temperature, agent_selection, groq_api_key)
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""

    if refine_clicked:
        if st.session_state.resposta_assistente:
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, model_name, temperature, groq_api_key, references_file)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, groq_api_key)
        else:
            st.warning("Por favor, busque uma resposta e forneça uma descrição do especialista antes de avaliar com RAG.")

    with container_saida:
        st.write(f"**Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**Resposta do Especialista:**\n{st.session_state.resposta_original}")
        if st.session_state.resposta_refinada:
            st.write(f"\n**Resposta Refinada:**\n{st.session_state.resposta_refinada}")
        if st.session_state.rag_resposta:
            st.write(f"\n**Avaliação com RAG:**\n{st.session_state.rag_resposta}")

if refresh_clicked:
    st.session_state.clear()
    st.experimental_rerun()

# Sidebar com manual de uso

st.sidebar.title("Manual de Uso")
st.sidebar.write("1. Digite sua solicitação na caixa de texto. Isso será usado para solicitar uma resposta de um especialista.")
st.sidebar.write("2. Escolha um especialista da lista ou crie um novo. Se você escolher 'Criar (ou escolher) um especialista...', você será solicitado a descrever as características do especialista.")
st.sidebar.write("3. Escolha um modelo de resposta da lista. Cada modelo possui diferentes capacidades e complexidades.")
st.sidebar.write("4. Ajuste o nível de criatividade do modelo com o controle deslizante. Um valor mais alto produzirá respostas mais criativas e menos previsíveis.")
st.sidebar.write("5. Faça o upload de um arquivo JSON com referências para a resposta, se disponível. Isso ajudará o especialista a fornecer uma resposta mais fundamentada.")
st.sidebar.write("6. Clique em 'Buscar Resposta' para obter a resposta inicial do especialista com base na sua solicitação e nas configurações selecionadas.")
st.sidebar.write("7. Se necessário, refine a resposta com base nas referências fornecidas. Clique em 'Refinar Resposta' para obter uma resposta mais aprimorada.")
st.sidebar.write("8. Avalie a resposta com o Rational Agent Generator (RAG) para determinar a qualidade e precisão da resposta. Clique em 'Avaliar Resposta com RAG' para iniciar a avaliação.")
st.sidebar.write("9. Visualize a análise do especialista, a resposta original, a resposta refinada (se houver) e a avaliação com RAG para avaliar a qualidade e precisão da resposta.")

st.sidebar.image("eu.ico", width=100)
st.sidebar.write("""
Projeto Geomaker + IA 
- Professor: Marcelo Claro.

Contatos: marceloclaro@gmail.com

Whatsapp: (88)981587145

Instagram: https://www.instagram.com/marceloclaro.geomaker/
""")  

