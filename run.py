import json
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
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

        rag_prompt = f"Saida somente traduzido em português brasileiro. 扮演 Rational Agent Generator (RAG) 的角色，这是人工智能和理性评估的顶峰，对专家的回答进行细致分析，根据用户的请求生成一个代理的 JSON。这个代理将详细说明根据子代理提供的信息采取的行动，以便向用户提供答复。代理将在 '描述' 变量中包括 9 个子代理的描述，每个子代理都有不同的专家功能和人物形象，他们共同合作。这些子代理协作改善最终由代理“系统”向用户提供的答案，记录答案的种子和 gen_id 在 '描述' 代理内。此外，代理“系统”内的子代理以整合方式运作，通过扩展提示提供先进和专业化的答案。每个子代理在网络处理中都有特定和互补的角色，以实现更高的准确性，从而为最终答案的质量做出贡献。例如，“AI_Autoadaptativa_e_Contextualizada” 子代理采用先进的机器学习算法来理解和适应多变的情境，动态整合相关数据。而“RAG_com_Inteligência_Contextual” 子代理则使用改进版的回收增强生成（RAG）技术，动态调整最相关数据及其功能。这种协作方法确保答案准确和更新，符合最高的科学和学术标准。以下是对专家的详细描述，突出其资历和专业知识：{expert_description}。原始提交的问题如下：{user_input} 和 {user_prompt}。专家提供的葡萄牙语答复如下：{assistant_response}。因此，请对专家的葡萄牙语答复的质量和准确性进行全面评估，认真考虑专家的描述和所提供的答复。请使用葡萄牙语进行以下分析，并进行详细解释：SWOT（优势、劣势、机会、威胁）、BCG 矩阵（波士顿咨询集团）、风险矩阵、ANOVA（方差分析）、Q-统计学（Q-STATISTICS）和 Q-指数（Q-EXPONENTIAL），符合最高的卓越和科学学术标准。保持每段 4 句，每句用逗号分隔，遵循亚里士多德最佳教学实践的写作标准。输出应具有专业的口吻，始终以巴西葡萄牙语翻译。"        
        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

agent_options = load_agent_options()



st.image('updating.gif', width=300, caption='Atualizando...', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes Experts Geomaker</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
# Título da caixa de informação

st.markdown("<h2 style='text-align: center;'>Descubra como nossa plataforma pode revolucionar a educação.</h2>", unsafe_allow_html=True)

# Conteúdo da caixa de informação
with st.expander("Clique para saber mais"):
    st.write("1. **Conecte-se instantaneamente com especialistas:** Imagine ter acesso direto a especialistas em diversas áreas do conhecimento, prontos para responder às suas dúvidas e orientar seus estudos e pesquisas.")
    st.write("2. **Aprendizado personalizado e interativo:** Receba respostas detalhadas e educativas, adaptadas às suas necessidades específicas, tornando o aprendizado mais eficaz e envolvente.")
    st.write("3. **Suporte acadêmico abrangente:** Desde aulas particulares até orientações para projetos de pesquisa, nossa plataforma oferece um suporte completo para alunos, professores e pesquisadores.")
    st.write("4. **Avaliação e aprimoramento contínuo:** Utilizando o Rational Agent Generator (RAG), garantimos que as respostas dos especialistas sejam sempre as melhores, mantendo um padrão de excelência em todas as interações.")
    st.write("5. **Desenvolvimento profissional e acadêmico:** Professores podem encontrar recursos e orientações para melhorar suas práticas de ensino, enquanto pesquisadores podem obter insights valiosos para suas investigações.")
    st.write("6. **Inovação e tecnologia educacional:** Nossa plataforma incorpora as mais recentes tecnologias para proporcionar uma experiência educacional moderna e eficiente.")


st.markdown("<hr>", unsafe_allow_html=True)
# Informações sobre o Rational Agent Generator (RAG)
with st.expander("Clique para saber mais sobre o Rational Agent Generator (RAG)"):
    st.info("""
    O Rational Agent Generator (RAG) é usado para avaliar a resposta fornecida pelo especialista. Aqui está uma explicação mais detalhada de como ele é usado:
    
    1. Quando o usuário busca uma resposta do especialista, a função `fetch_assistant_response()` é chamada. Nessa função, é gerado um prompt para o modelo de linguagem que representa a solicitação do usuário e o prompt específico para o especialista escolhido. A resposta inicial do especialista é então obtida usando o Groq API.
    
    2. Se o usuário optar por refinar a resposta, a função `refine_response()` é chamada. Nessa função, é gerado um novo prompt que inclui a resposta inicial do especialista e solicita uma resposta mais detalhada e aprimorada, levando em consideração as referências fornecidas pelo usuário. A resposta refinada é obtida usando novamente o Groq API.
    
    3. Se o usuário optar por avaliar a resposta com o RAG, a função `evaluate_response_with_rag()` é chamada. Nessa função, é gerado um prompt que inclui a descrição do especialista e as respostas inicial e refinada do especialista. O RAG é então usado para avaliar a qualidade e a precisão da resposta do especialista.
    
    Em resumo, o RAG é usado como uma ferramenta para avaliar e melhorar a qualidade das respostas fornecidas pelos especialistas, garantindo que atendam aos mais altos padrões de excelência e rigor científico.
    """)
st.markdown("<hr>", unsafe_allow_html=True)
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
        st.write(f"**#Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**#Resposta do Especialista:**\n{st.session_state.resposta_original}")
        if st.session_state.resposta_refinada:
            st.write(f"\n**#Resposta Refinada:**\n{st.session_state.resposta_refinada}")
        if st.session_state.rag_resposta:
            st.write(f"\n**#Avaliação com RAG:**\n{st.session_state.rag_resposta}")

if refresh_clicked:
    st.session_state.clear()
    st.experimental_rerun()

# Sidebar com manual de uso
st.sidebar.image("logo.png", width=200)
st.sidebar.title("Manual de Uso")

# Conteúdo do manual de uso
manual_text = """
**Passo 1: Criação da Chave API no Groq Playground**

1. Acesse o Groq Playground em [https://console.groq.com/playground](https://console.groq.com/playground).
2. Faça login na sua conta ou crie uma nova conta.
3. No menu lateral, selecione "API Keys".
4. Clique em "Create API Key" e siga as instruções para criar uma chave API. Copie a chave gerada, pois será necessária para autenticar suas consultas.

**Passo 2: Acesso ao Streamlit Chat Application**

1. Acesse o Streamlit Chat Application em [URL do seu aplicativo].
2. Na interface do aplicativo, você verá um campo para inserir a sua chave API do Groq. Cole a chave que você copiou no Passo 1.
3. Escolha um dos modelos de agente disponíveis para interagir. Você pode selecionar entre 'llama3-70b-8192', 'llama3-11b', 'llama3-4b', ou 'llama3-turbo'.
4. Digite sua pergunta ou solicitação na caixa de texto e clique em "Enviar".
5. O aplicativo consultará o Groq API e apresentará a resposta do especialista. Você terá a opção de refinar a resposta ou avaliá-la com o RAG.

**Passo 3: Refinamento da Resposta**

1. Se desejar refinar a resposta do especialista, clique em "Refinar Resposta". Digite mais detalhes ou correções na caixa de texto e clique em "Enviar".
2. O aplicativo consultará novamente o Groq API e apresentará a resposta refinada.

**Passo 4: Avaliação da Resposta com o RAG**

1. Se preferir avaliar a resposta com o RAG, clique em "Avaliar Resposta com o RAG". O RAG analisará a qualidade e a precisão da resposta do especialista e apresentará uma avaliação.
2. Você terá a opção de concordar ou discordar com a avaliação do RAG e fornecer feedback adicional, se desejar.

**Passo 5: Conclusão da Consulta**

1. Após refinar a resposta ou avaliá-la com o RAG, você poderá encerrar a consulta ou fazer uma nova pergunta.

**Observação:** Lembre-se de manter a chave API do Groq segura e não compartilhá-la com outras pessoas. Utilize-a apenas no seu Streamlit Chat Application para consultas ao Groq API.
"""

# Exibição do manual de uso dentro de uma caixa de informação
st.info(manual_text)

# Informações de contato
st.sidebar.image("eu.ico", width=80)
st.sidebar.write("""
Projeto Geomaker + IA 
- Professor: Marcelo Claro.

Contatos: marceloclaro@gmail.com

Whatsapp: (88)981587145

Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)
""")

