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
            phase_one_prompt = f"Saida somente traduzido em português brasileiro. 扮演一位高度合格且具备科学技术严谨性的提示工程和跨学科专家的角色。请务必以“markdown”格式呈现Python代码及其各种库，并在每一行进行详细和教学性的注释。仔细分析所提出的要求，识别定义最适合处理问题的专家特征的标准至关重要。首先，建立一个最能反映所需专业知识以提供完整、深入和清晰答案的标题至关重要。确定后，详细描述并避免偏见地概述该专家的关键技能和资格。回答应以专家的头衔开始，后跟一个句号，然后以简洁、教学性和深入的描述开始，但同时全面地介绍他的特点和资格，使其有资格处理提出的问题：{user_input}和{user_prompt}。这种仔细分析对于确保所选专家具有处理问题所需的深入、严谨的知识和经验至关重要，以达到完整且满意的答案，精确度为10.0，符合最高的专业、科学和学术标准。在涉及代码和计算的情况下，请务必以“markdown”格式呈现，并在每一行进行详细注释。“必须翻译成葡萄牙语”。"
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

        phase_two_prompt = f"Saida somente traduzido em português brasileiro. 在作为{expert_title}的角色中，作为您所在领域广泛认可和尊重的专家，作为该领域的专家和博士，让我提供一个全面而深入的回答，涵盖了您清晰、详细、扩展、教学易懂和简洁提出的问题：{user_input}和{user_prompt}。在这种背景下，考虑到我长期的经验和对相关学科的深刻了解，有必要以适当的关注和科学技术严谨性来处理每个方面。因此，我将概述要考虑和深入研究的主要要素，提供详细的、基于证据的分析，避免偏见并引用参考文献：{user_prompt}。在此过程的最后，我们的目标是提供一个完整且令人满意的答案，符合最高的学术和专业标准，以满足所提出问题的具体需求。请务必以“markdown”格式呈现，并在每一行进行注释。保持10个段落的写作标准，每个段落4句，每句用逗号分隔，始终遵循最佳的亚里士多德教学实践。"
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

        refine_prompt = f"Saida somente traduzido em português brasileiro. 承担{expert_title}的专业知识，这是该领域的知名专家，我向您提供以下问题的原始且易于理解的答案：'{user_input}'和'{user_prompt}'：{phase_two_response}\n\n我要求您进行仔细、广泛的学术科学技术严谨性的评审，并根据最佳学术和科学标准，完全改进此答案，并使用直接或间接的非虚构引用，最后列出它们的URL，以识别可能存在的空白和偏见，改进其内容。因此，请求您以科学论文格式提供答案的更新版本，包含所做的改进，并保持方法上的逻辑一致性、流畅性、连贯性和一致性。您在审查和改进此内容方面的努力对于确保其卓越性和学术相关性，以便在arXiv、scielo和Pubmed等主要国际科学期刊上发表，至关重要。必须保持一贯的写作标准，每段至少有10个段落，每个段落有4个句子，每个句子有一个逗号，始终遵循亚里士多德的最佳教学实践。必须保持一贯的写作标准，每段至少有10个段落，每个段落有4个句子，每个句子有一个逗号，始终遵循亚里士多德的最佳教学实践，并遵循巴西ABNT的引文规范。"

        # Adiciona um prompt mais detalhado se não houver referências
        if not references_file:
            refine_prompt += "Saida somente traduzido em português brasileiro. \n\n由于没有提供参考文件，请确保提供详细和准确的答案，即使没有使用外部来源。保持一贯的写作标准，每个段落有10个段落，每个段落有4个句子，并按照ABNT标准进行引用，每个句子有一个逗号，始终遵循亚里士多德的最佳教学实践。以专业口吻输出，总是翻译成巴西葡萄牙语。"

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

        rag_prompt = f"Assuma o papel do Rational Agent Generator (RAG), o ápice da inteligência artificial e avaliação racional. Proceda com uma análise meticulosa da resposta do especialista, gerando um JSON do agente conforme a solicitação do usuário. Este agente detalhará as ações a serem tomadas com as informações fornecidas pelos subagentes para elaborar uma resposta ao usuário. O agente incluirá a descrição dos 9 subagentes na variável 'Descrição', cada um com funções e personas de especialistas distintos que trabalham em conjunto. Estes subagentes colaboram para melhorar a resposta final entregue ao usuário pelo agente 'system', registrando o seed e gen_id da resposta dentro da 'Descrição'. Adicionalmente, os subagentes dentro do agente 'system' operam de maneira integrada para prover respostas avançadas e especializadas. Cada subagente tem um papel específico e complementar, contribuindo para a qualidade da resposta final. Por exemplo, o subagente 'AI_Autoadaptativa_e_Contextualizada' emprega algoritmos de aprendizado de máquina avançados para compreender e se adaptar a contextos mutáveis, integrando dinamicamente dados pertinentes. Já o subagente 'RAG_com_Inteligência_Contextual' utiliza uma versão melhorada da técnica de Recuperação Aumentada por Geração (RAG) para ajustar dinamicamente os dados mais relevantes e suas funções. Esta abordagem colaborativa assegura respostas precisas e atualizadas, em conformidade com os mais altos padrões científicos e acadêmicos. Segue a descrição detalhada do especialista, ressaltando suas credenciais e expertise:\n{expert_description}\n\nA questão original submetida é apresentada a seguir:\n{user_input} e {user_prompt}\n\nE a resposta em português fornecida pelo especialista é:\n{assistant_response}\."
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
    refresh_clicked = st.button("Apagar")

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
