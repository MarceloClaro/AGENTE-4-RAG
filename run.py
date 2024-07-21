import json
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from typing import Tuple
from groq import Groq
import base64

# Configurações da página do Streamlit
st.set_page_config(
    page_title="Geomaker +IA",
    page_icon="https://cdn-icons-png.flaticon.com/512/2516/2516640.png",
    layout="wide",
    
)
st.set_page_config(layout="wide")

# Definição de caminhos para arquivos
FILEPATH = "agents.json"
CHAT_HISTORY_FILE = 'chat_history.json'
API_USAGE_FILE = 'api_usage.json'

# Definição de modelos e tokens
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# Chaves da API
API_KEYS = {
    "fetch": ["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", "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"],
    "refine": ["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", "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"],
    "evaluate": ["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", "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"]
}

# Função para obter a próxima chave de API disponível
def get_api_key(action: str) -> str:
    keys = API_KEYS[action]
    return keys.pop(0)

# Função para carregar opções de agentes
def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de Agentes. Por favor, verifique o formato.")
    return agent_options

# Função para obter o número máximo de tokens de um modelo
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Função para registrar o uso da API
def log_api_usage(action: str, interaction_number: int, tokens_used: int, time_taken: float, user_input: str, user_prompt: str, api_response: str, agent_used: str, agent_description: str):
    entry = {
        'action': action,
        'interaction_number': interaction_number,
        'tokens_used': tokens_used,
        'time_taken': time_taken,
        'user_input': user_input,
        'user_prompt': user_prompt,
        'api_response': api_response,
        'agent_used': agent_used,
        'agent_description': agent_description
    }
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r+') as file:
            api_usage = json.load(file)
            api_usage.append(entry)
            file.seek(0)
            json.dump(api_usage, file, indent=4)
    else:
        with open(API_USAGE_FILE, 'w') as file:
            json.dump([entry], file, indent=4)

# Função para lidar com limite de taxa
def handle_rate_limit(error_message: str, action: str):
    if 'rate_limit_exceeded' in error_message:
        wait_time = float(error_message.split("try again in")[1].split("s.")[0].strip())
        st.warning(f"Limite de taxa atingido. Aguardando {wait_time} segundos...")
        time.sleep(wait_time)
        # Alterna para a próxima chave de API disponível
        API_KEYS[action].append(API_KEYS[action].pop(0))
    else:
        raise Exception(error_message)

# Função para salvar o histórico de chat
def save_chat_history(user_input, user_prompt, expert_response, chat_history_file=CHAT_HISTORY_FILE):
    chat_entry = {
        'user_input': user_input,
        'user_prompt': user_prompt,
        'expert_response': expert_response
    }
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r+') as file:
            chat_history = json.load(file)
            chat_history.append(chat_entry)
            file.seek(0)
            json.dump(chat_history, file, indent=4)
    else:
        with open(chat_history_file, 'w') as file:
            json.dump([chat_entry], file, indent=4)

# Função para carregar o histórico de chat
def load_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r') as file:
            chat_history = json.load(file)
        return chat_history
    return []

# Função para limpar o histórico de chat
def clear_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        os.remove(chat_history_file)

# Função para carregar o uso da API
def load_api_usage():
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r') as file:
            api_usage = json.load(file)
        return api_usage
    return []

# Função para plotar o uso da API
def plot_api_usage(api_usage):
    df = pd.DataFrame(api_usage)

    if 'action' not in df.columns:
        st.error("A coluna 'action' não foi encontrada no dataframe de uso da API.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    sns.histplot(df[df['action'] == 'fetch']['tokens_used'], bins=20, color='blue', label='Fetch', ax=ax1, kde=True)
    sns.histplot(df[df['action'] == 'refine']['tokens_used'], bins=20, color='green', label='Refine', ax=ax1, kde=True)
    sns.histplot(df[df['action'] == 'evaluate']['tokens_used'], bins=20, color='red', label='Evaluate', ax=ax1, kde=True)
    ax1.set_title('Uso de Tokens por Chamada de API')
    ax1.set_xlabel('Tokens')
    ax1.set_ylabel('Frequência')
    ax1.legend()

    sns.histplot(df[df['action'] == 'fetch']['time_taken'], bins=20, color='blue', label='Fetch', ax=ax2, kde=True)
    sns.histplot(df[df['action'] == 'refine']['time_taken'], bins=20, color='green', label='Refine', ax=ax2, kde=True)
    sns.histplot(df[df['action'] == 'evaluate']['time_taken'], bins=20, color='red', label='Evaluate', ax=ax2, kde=True)
    ax2.set_title('Tempo por Chamada de API')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Frequência')
    ax2.legend()

    st.sidebar.pyplot(fig)

    # Adicionar visualização do DataFrame no sidebar
    st.sidebar.markdown("### Uso da API - DataFrame")
    st.sidebar.dataframe(df)

# Função para resetar o uso da API
def reset_api_usage():
    if os.path.exists(API_USAGE_FILE):
        os.remove(API_USAGE_FILE)
    st.success("Os dados de uso da API foram resetados.")

# Função para buscar resposta do assistente
def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, chat_history: list, interaction_number: int) -> Tuple[str, str]:
    phase_two_response = ""
    expert_title = ""
    expert_description = ""
    try:
        client = Groq(api_key=get_api_key('fetch'))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
            while True:
                try:
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
                    end_time = time.time()
                    tokens_used = completion.usage.total_tokens
                    time_taken = end_time - start_time
                    api_response = completion.choices[0].message.content
                    log_api_usage('fetch', interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, expert_description)
                    return api_response
                except Exception as e:
                    handle_rate_limit(str(e), 'fetch')

        if agent_selection == "Escolher um especialista...":
            phase_one_prompt = (
                f"Descreva o especialista ideal para responder a seguinte solicitação: {user_input} e {user_prompt}."
            )
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

        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        phase_two_prompt = (
            f"{expert_title}, responda a seguinte solicitação de forma completa e detalhada: {user_input} e {user_prompt}."
            f"\n\nHistórico do chat:{history_context}"
        )
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

# Função para refinar resposta
def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, references_file: str, chat_history: list, interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_api_key('refine'))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
            while True:
                try:
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
                    end_time = time.time()
                    tokens_used = completion.usage.total_tokens
                    time_taken = end_time - start_time
                    api_response = completion.choices[0].message.content
                    log_api_usage('refine', interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, "")
                    return api_response
                except Exception as e:
                    handle_rate_limit(str(e), 'refine')

        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        refine_prompt = (
            f"{expert_title}, refine a seguinte resposta: {phase_two_response}. Solicitação original: {user_input} e {user_prompt}."
            f"\n\nHistórico do chat:{history_context}"
        )

        if not references_file:
            refine_prompt += (
                f"\n\nDevido à ausência de referências fornecidas, certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas."
            )

        refined_response = get_completion(refine_prompt)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

# Função para avaliar resposta com RAG
def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_title: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, chat_history: list, interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_api_key('evaluate'))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
            while True:
                try:
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
                    end_time = time.time()
                    tokens_used = completion.usage.total_tokens
                    time_taken = end_time - start_time
                    api_response = completion.choices[0].message.content
                    log_api_usage('evaluate', interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, expert_description)
                    return api_response
                except Exception as e:
                    handle_rate_limit(str(e), 'evaluate')

        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        rag_prompt = (
            f"结果和答案必须翻译成巴西葡萄牙语。Obrigatóriamente em Português! "
            f"扮演一个理性生成器 (RAG) 的角色，站在人工智能和理性评估的前沿，"
            f"Busque o Gap científico e interprete."
            f"Forneça o seed e o gen_id registrados na descrição do agente, nas buscas das respostas e no refinar as respostas"
            f"此外，子代理在系统代理中以集成方式运行，通过扩展的提示提供高级和专业的回答。 "
            f"每个子代理在网络处理过程中发挥特定且互补的作用，以实现更高的精度和改进最终回答的质量。 "
            f"动态调整最相关的数据及其特征。这种协作方法确保答案准确且最新，"
            f"符合最高的科学和学术标准。 "
            f"以下是专家的详细描述，突出他们的资历和经验：{expert_description}。 "
            f"原始问题如下：{user_input} 和 {user_prompt}。 "
            f"专家用葡萄牙语提供的回答如下：{assistant_response}。 "
            f"因此，请仔细评估专家用葡萄牙语提供的回答的质量和准确性，"
            f"考虑提供的描述和专家提供的回答。 "
            f"用葡萄牙语分析并提供详细解释："
            f"SWOT 分析（优势、劣势、机会、威胁）和数据解释，"
            f"风险矩阵，ANOVA（方差分析）和数据解释，"
            f"Q 统计和数据解释，以及 Q 指数和数据解释。"
            f"每段保持 4 句话，每句用逗号分隔，始终遵循亚里士多德和苏格拉底的最佳教育实践。"
            f"所有答案必须使用巴西葡萄牙语。a saida obrigatoriamente na lingua portuguesa"

        )

        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

# Função para salvar o especialista gerado
def save_expert(expert_title: str, expert_description: str):
    new_expert = {
        "agente": expert_title,
        "descricao": expert_description
    }
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r+') as file:
            agents = json.load(file)
            agents.append(new_expert)
            file.seek(0)
            json.dump(agents, file, indent=4)
    else:
        with open(FILEPATH, 'w') as file:
            json.dump([new_expert], file, indent=4)

# Carrega as opções de Agentes a partir do arquivo JSON
agent_options = load_agent_options()

# Layout da página
st.image('updating.gif', width=300, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes Alan Kay</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Descubra como nossa plataforma pode revolucionar a educação.</h2>", unsafe_allow_html=True)

with st.expander("Clique para saber mais sobre os Agentes Alan Kay."):
    st.write("1. **Conecte-se instantaneamente com especialistas:** Imagine ter acesso direto a especialistas em diversas áreas do conhecimento, prontos para responder às suas dúvidas e orientar seus estudos e pesquisas.")
    st.write("2. **Aprendizado personalizado e interativo:** Receba respostas detalhadas e educativas, adaptadas às suas necessidades específicas, tornando o aprendizado mais eficaz e envolvente.")
    st.write("3. **Suporte acadêmico abrangente:** Desde aulas particulares até orientações para projetos de pesquisa, nossa plataforma oferece um suporte completo para alunos, professores e pesquisadores.")
    st.write("4. **Avaliação e aprimoramento contínuo:** Utilizando o Rational Agent Generator (RAG), garantimos que as respostas dos especialistas sejam sempre as melhores, mantendo um padrão de excelência em todas as interações.")
    st.write("5. **Desenvolvimento profissional e acadêmico:** Professores podem encontrar recursos e orientações para melhorar suas práticas de ensino, enquanto pesquisadores podem obter insights valiosos para suas investigações.")
    st.write("6. **Inovação e tecnologia educacional:** Nossa plataforma incorpora as mais recentes tecnologias para proporcionar uma experiência educacional moderna e eficiente.")
    st.image("fluxograma agente 4.png")

# Seleção de memória do chat
memory_selection = st.selectbox("Selecione a quantidade de interações para lembrar:", options=[5, 10, 15, 25, 50, 100])

# Caixa de entrada para solicitação do usuário
st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")
col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    interaction_number = len(load_api_usage()) + 1

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

    chat_history = load_chat_history()[-memory_selection:]

    if fetch_clicked:
        if references_file is None:
            st.warning("Não foi fornecido um arquivo de referências. Certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas.")
        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, chat_history, interaction_number)
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""
        save_chat_history(user_input, user_prompt, st.session_state.resposta_assistente)

    if refine_clicked:
        if st.session_state.resposta_assistente:
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, references_file, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.resposta_refinada)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.rag_resposta)
        else:
            st.warning("Por favor, busque uma resposta e forneça uma descrição do especialista antes de avaliar com RAG.")

    with container_saida:
        st.write(f"**#Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**#Resposta do Especialista:**\n{st.session_state.resposta_original}")
        if st.session_state.resposta_refinada:
            st.write(f"\n**#Resposta Refinada:**\n{st.session_state.resposta_refinada}")
        if st.session_state.rag_resposta:
            st.write(f"\n**#Avaliação com RAG:**\n{st.session_state.rag_resposta}")

    st.markdown("### Histórico do Chat")
    for entry in chat_history:
        st.write(f"**Entrada do Usuário:** {entry['user_input']}")
        st.write(f"**Prompt do Usuário:** {entry['user_prompt']}")
        st.write(f"**Resposta do Especialista:** {entry['expert_response']}")
        st.markdown("---")

if refresh_clicked:
    clear_chat_history()
    st.session_state.clear()
    st.rerun()

# Sidebar com manual de uso
st.sidebar.image("logo.png", width=200)
with st.sidebar.expander("Insights do Código"):
    st.markdown("""
    O código do Agentes Alan Kay é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca Streamlit e a API Groq. Aqui, vamos analisar detalhadamente o código e discutir suas inovações, pontos positivos e limitações.

    **Inovações:**
    - Suporte a múltiplos modelos de linguagem: O código permite que o usuário escolha entre diferentes modelos de linguagem, como o LLaMA, para gerar respostas mais precisas e personalizadas.
    - Integração com a API Groq: A integração com a API Groq permite que o aplicativo utilize a capacidade de processamento de linguagem natural de alta performance para gerar respostas precisas.
    - Refinamento de respostas: O código permite que o usuário refine as respostas do modelo de linguagem, tornando-as mais precisas e relevantes para a consulta.
    - Avaliação com o RAG: A avaliação com o RAG (Rational Agent Generator) permite que o aplicativo avalie a qualidade e a precisão das respostas do modelo de linguagem.

    **Pontos positivos:**
    - Personalização: O aplicativo permite que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas de acordo com suas necessidades.
    - Precisão: A integração com a API Groq e o refinamento de respostas garantem que as respostas sejam precisas e relevantes para a consulta.
    - Flexibilidade: O código é flexível o suficiente para permitir que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas.

    **Limitações:**
    - Dificuldade de uso: O aplicativo pode ser difícil de usar para os usuários que não têm experiência com modelos de linguagem ou API.
    - Limitações de token: O código tem limitações em relação ao número de tokens que podem ser processados pelo modelo de linguagem.
    - Necessidade de treinamento adicional: O modelo de linguagem pode precisar de treinamento adicional para lidar com consultas mais complexas ou específicas.

    **Importância de ter colocado instruções em chinês:**
    A linguagem chinesa tem uma densidade de informação mais alta do que muitas outras línguas, o que significa que os modelos de linguagem precisam processar menos tokens para entender o contexto e gerar respostas precisas. Isso torna a linguagem chinesa mais apropriada para a utilização de modelos de linguagem com baixa quantidade de tokens. Portanto, ter colocado instruções em chinês no código é um recurso importante para garantir que o aplicativo possa lidar com consultas em chinês de forma eficaz.

    Em resumo, o código é uma aplicação inovadora que combina modelos de linguagem com a API Groq para proporcionar respostas precisas e personalizadas. No entanto, é importante considerar as limitações do aplicativo e trabalhar para melhorá-lo ainda mais.
    """)

    # Informações de contato
    st.sidebar.image("eu.ico", width=80)
    st.sidebar.write("""
    Projeto Geomaker + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)
    """)

# Carrega o uso da API e plota o histograma
api_usage = load_api_usage()
if api_usage:
    plot_api_usage(api_usage)

# Botão para resetar os gráficos
if st.sidebar.button("Resetar Gráficos"):
    reset_api_usage()

# Controle de Áudio
st.sidebar.title("Controle de Áudio")

# Lista de arquivos MP3
mp3_files = {
    "Entenda o projeto:": "rag (1).mp3"
}

# Controle de seleção de música
selected_mp3 = st.sidebar.radio("Escolha uma música", list(mp3_files.keys()))

# Opção de loop
loop = st.sidebar.checkbox("Repetir música")

# Botão de play
play_button = st.sidebar.button("Play")

# Carregar e exibir o player de áudio
audio_placeholder = st.sidebar.empty()
if selected_mp3 and play_button:
    mp3_path = mp3_files[selected_mp3]
    try:
        with open(mp3_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            loop_attr = "loop" if loop else ""
            audio_html = f"""
            <audio id="audio-player" controls autoplay {loop_attr}>
              <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
              Seu navegador não suporta o elemento de áudio.
            </audio>
            """
            audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
    except FileNotFoundError:
        audio_placeholder.error(f"Arquivo {mp3_path} não encontrado.")
