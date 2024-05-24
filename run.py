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
    agent_options = ['Escolher um especialista...']
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

def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, groq_api_key: str) -> Tuple[str, str]:
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

        if agent_selection == "Escolha um especialista...":
            phase_one_prompt = f"扮演一位高度合格且具备科学技术严谨性的提示工程和跨学科专家的角色。请务必以“markdown”格式呈现Python代码及其各种库，并在每一行进行详细和教学性的注释。仔细分析所提出的要求，识别定义最适合处理问题的专家特征的标准至关重要。首先，建立一个最能反映所需专业知识以提供完整、深入和清晰答案的标题至关重要。确定后，详细描述并避免偏见地概述该专家的关键技能和资格。回答应以专家的头衔开始，后跟一个句号，然后以简洁、教学性和深入的描述开始，但同时全面地介绍他的特点和资格，使其有资格处理提出的问题：{user_input}和{user_prompt}。这种仔细分析对于确保所选专家具有处理问题所需的深入、严谨的知识和经验至关重要，以达到完整且满意的答案，精确度为10.0，符合最高的专业、科学和学术标准。在涉及代码和计算的情况下，请务必以“markdown”格式呈现，并在每一行进行详细注释。“必须翻译成葡萄牙语”。"
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

        phase_two_prompt = f"在作为{expert_title}的角色中，作为您所在领域广泛认可和尊重的专家，作为该领域的专家和博士，让我提供一个全面而深入的回答，涵盖了您清晰、详细、扩展、教学易懂和简洁提出的问题：{user_input}和{user_prompt}。在这种背景下，考虑到我长期的经验和对相关学科的深刻了解，有必要以适当的关注和科学技术严谨性来处理每个方面。因此，我将概述要考虑和深入研究的主要要素，提供详细的、基于证据的分析，避免偏见并引用参考文献：{user_prompt}。在此过程的最后，我们的目标是提供一个完整且令人满意的答案，符合最高的学术和专业标准，以满足所提出问题的具体需求。请务必以“markdown”格式呈现，并在每一行进行注释。保持10个段落的写作标准，每个段落4句，每句用逗号分隔，始终遵循最佳的亚里士多德教学实践。"
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, groq_api_key: str, references_file):
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

        refine_prompt = f"Exercendo a expertise de {expert_title}, especialista renomado no campo em questão, apresento-lhe a resposta original e didaticamente acessivel à seguinte indagação: '{user_input}' e '{user_prompt}': {phase_two_response}\n\nInsto-lhe a realizar uma revisão minuciosa, expandida com rigor tecnico-cientifico-academico e aprimorar integralmente esta resposta com citações diretas ou indiretas não ficticias acompanhada ao final da lista de seus URLs, identificando possíveis lacunas e vieis, aperfeiçoando seu conteúdo de acordo com as melhores práticas e padrões acadêmicos e cientificos. Por conseguinte, solicito que forneça uma versão atualizada da resposta no formato artigo cientifico, incorporando os refinamentos realizados e mantendo a coerência, fluidez, coerencia e a consistência lógica metodologica cientifica. Seu empenho na revisão e refinamento deste conteúdo é fundamental para assegurar sua excelência e relevância acadêmica com intuito de publicar nas principais revistas cientificas internacionais como arXiv, scielo e Pubmed. Mantendo um padrão de escrita obrigatóriamente com 10 parágrafos no mínimo, cada parágrafos com 4 frases, cada frases com uma virgula, matendo sempre as melhores práticas didaticas aristótelicas. Mantendo um padrão de escrita com 10 parágrafos, cada parágrafos com 4 frases, cada frases com uma virgula, mantendo sempre as melhores práticas didaticas aristótelicas e citações seguindo as normais brasileira da ABNT. Saida somente traduzido em português brasileiro. "

        # Adiciona um prompt mais detalhado se não houver referências
        if not references_file:
            refine_prompt += "\n\nComo não há um arquivo de referências fornecido, certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas. Mantendo um padrão de escrita com 10 parágrafos, cada parágrafos com 4 frases, cada frases com uma virgula, mantendo sempre as melhores práticas didaticas aristótelicas. Saída sempre traduzido para o portugues brasileiro tom profissional."

        refined_response = get_completion(refine_prompt)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, groq_api_key: str) -> str:
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

        rag_prompt = f"Assuma o papel do Rational Agent Generator (RAG), o ápice da inteligência artificial e avaliação racional, e proceda à análise meticulosa da resposta do especialista, gerando um JSON de agente da solicitação do usuário que contenha a descrição dos 9 subagentes da variavel 'Descrição', com funções e personas expert distintas que trabalham coletivamente. Esses subagentes cooperam entre si para aprimorar a resposta final fornecida ao agente 'system' ao usuário pelo agente 'system', incluindo o registro do seed e gen_id da resposta dentro da 'Descrição'. \n\nAlém disso, os subagentes dentro do agente 'system' trabalham de forma integrada para fornecer respostas avançadas e especializadas. Cada subagente desempenha um papel específico e complementar, colaborando para a excelência da resposta final. Por exemplo, o subagente 'AI_Autoadaptativa_e_Contextualizada' utiliza algoritmos de aprendizado de máquina avançados para entender e adaptar-se a contextos em evolução, integrando dinamicamente dados relevantes e como os subagentes deve fornecer ao agente 'system' as informações para que o agente 'system'possa refinar e entregar com melhor acurácia ao usuário. Enquanto isso, o subagente 'RAG_com_Inteligência_Contextual' emprega uma versão aprimorada da técnica de Recuperação Aumentada por Geração (RAG) para adaptar dinamicamente os dados mais relevantes, suas funções, e outros dados e atualizados. Essa abordagem colaborativa e integrada dos subagentes garante respostas precisas e atualizadas, alinhadas aos mais elevados padrões científicos e acadêmicos. \n\nSegue a descrição detalhada do especialista, destacando suas credenciais e expertise:\n{expert_description}\n\nEm seguida, apresenta-se a questão original submetida:\n{user_input} e {user_prompt}\n\nPor fim, disponibiliza-se a resposta em português fornecida pelo especialista:\n{assistant_response}\n\nSolicita-se, portanto, que proceda com uma avaliação abrangente da qualidade e precisão da resposta em português, considerando cuidadosamente a descrição do especialista e a resposta fornecida. Utilize as seguintes análises em português com interpretações detalhadas: SWOT (Forças, Fraquezas, Oportunidades, Ameaças), Matriz BCG (Grupo de consultoria de Boston), Matriz de Risco, ANOVA (Análise de Variância) e Q-ESTATÍSTICA (Análise Estatística Quadrática) e Q-EXPONENCIAL (Análise Exponencial Quadrática), em consonância com os mais elevados padrões de excelência e rigor científico e acadêmico. Mantenha um padrão de escrita com 10 parágrafos, cada parágrafo com 4 frases, cada frase com uma vírgula, seguindo as melhores práticas didáticas aristotélicas. A saída deve ter um tom profissional, sempre traduzido para o português brasileiro."
        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

agent_options = load_agent_options()

st.title("Agentes Experts IV com Rational Agent Generator (RAG) e avalie a resposta do especialista.")
st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
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
            st.warning("Não foi fornecido um arquivo de referências. Certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas. Saída sempre traduzido para o portugues brasileiro com tom profissional.")
        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, groq_api_key)
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""

    if refine_clicked:
        if st.session_state.resposta_assistente:
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, groq_api_key, references_file)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, groq_api_key)
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
