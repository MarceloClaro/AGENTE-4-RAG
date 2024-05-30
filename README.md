### AGENTE-4 RAG

AGENTE-4 RAG é um aplicativo web desenvolvido com Streamlit para fornecer respostas especializadas utilizando modelos de linguagem treinados pela API Groq.

---

#### Funções Principais

- **load_agent_options**: Carrega opções de especialistas do arquivo `agents.json`.
- **get_max_tokens**: Retorna o número máximo de tokens para um modelo específico.
- **fetch_assistant_response**: Obtém a resposta do especialista para uma pergunta do usuário.
- **refine_response**: Refina a resposta do especialista com base em referências fornecidas.
- **evaluate_response_with_rag**: Avalia a resposta do especialista usando RAG (Retrieval-Augmented Generation).

---

#### Como Usar

1. **Entrada de Texto**: Digite sua pergunta na área de texto.
2. **Escolha de Especialista**: Selecione o especialista desejado.
3. **Escolha de Modelo**: Selecione o modelo de linguagem.
4. **Nível de Criatividade**: Ajuste o controle deslizante para definir a criatividade da resposta.
5. **Chave da API**: Insira sua chave de API Groq.
6. **Botões**:
   - **Buscar Resposta**: Obtém a resposta do especialista.
   - **Refinar Resposta**: Refina a resposta usando referências.
   - **Atualizar Página**: Redefine a interface.
7. **Upload de Referências**: Faça upload de arquivos JSON para fornecer referências adicionais.

---

#### Inovações

- **Interface Intuitiva**: Utiliza Streamlit para criar uma interface interativa e fácil de usar.
- **Modelos Personalizáveis**: Permite selecionar diferentes modelos e ajustar a criatividade.
- **Integração com API Groq**: Tira proveito dos poderosos modelos de linguagem da API Groq para respostas precisas e contextualizadas.
- **Avaliação com RAG**: Avalia e melhora as respostas usando a técnica RAG, combinando recuperação de informações com geração de texto.

---

### Ilustrações

#### Diagrama de Funcionamento
![Diagrama de Funcionamento](https://raw.githubusercontent.com/MarceloClaro/AGENTE-4-RAG/main/diagram%20agente%204.png)

#### Fluxograma Manual
![Fluxograma Manual](https://raw.githubusercontent.com/MarceloClaro/AGENTE-4-RAG/main/fluxograma%20manual%20agente%204.png)

---

Para mais detalhes, visite o [AGENTE-4 RAG no Streamlit](https://agente4.streamlit.app/).
