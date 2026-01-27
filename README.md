# 🤖 Agente LangGraph com Qdrant e Geração de PDF

Este projeto contém um agente inteligente construído com **LangGraph**, utilizando **Qdrant** para memória vetorial e **WeasyPrint** para gerar documentos em PDF.

---

## 🚀 Guia de Configuração e Execução

Siga os passos abaixo para configurar as dependências de sistema, variáveis de ambiente e iniciar o servidor de desenvolvimento.

### 1.1 Variáveis de Ambiente (.env)

Configure as chaves de API e as configurações de rastreamento (LangSmith) essenciais para o funcionamento do agente.

1. Crie um arquivo chamado `.env` na raiz do projeto (baseado no `example.env` se existir).
2. Preencha com suas credenciais:

```env
# API Keys
GOOGLE_API_KEY=your_google_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here

# Configuração de Email para envio de PDFs (Gmail)
# IMPORTANTE: Use uma "Senha de App" do Gmail, não sua senha normal
# Veja EMAIL_SETUP.md para instruções detalhadas de como gerar
EMAIL_PASSWORD=your_gmail_app_password_here
REMETENTE=seu_email@gmail.com
QDRANT_URL = "https://your_qdrant_instance_url_here"
QDRANT_API_KEY = "your_qdrant_api_key_here"
```

### 1.2. Configuração do Qdrant (Vector Database)

Este projeto utiliza o Qdrant para armazenamento de memória vetorial.

1. Crie uma conta gratuita no [Qdrant Cloud](https://cloud.qdrant.io/).
2. Crie um novo **Cluster** (o tier gratuito é suficiente) Com nome **Tide**.
3. Vá em **Data Access Control** e gere uma nova **API Key**.
4. Copie a **URL** do cluster e a **API Key** para o seu arquivo `.env` (como mostrado acima).

---

### 2. Instalação de Dependências (Python)

Instale os pacotes listados no `requirements.txt` utilizando o gerenciador de sua preferência (`uv` ou `pip`).

**Usando uv:**

```bash
uv pip install -r requirements.txt
```

**Usando pip:**

```bash
pip install -r requirements.txt
```

---

### ⚠️ 3. Configuração de PDF (Apenas Windows)

O pacote **WeasyPrint** requer bibliotecas nativas GLib/GTK no Windows (`libgobject-2.0-0.dll`). Se você estiver recebendo erros de DLL, siga estas etapas obrigatórias:

1. **Instale o MSYS2**: Baixe e instale o [MSYS2](https://www.msys2.org/).
2. **Instale o GTK**: Abra o terminal **MSYS2 MinGW 64-bit** e execute os comandos abaixo:

    ```bash
    # 1. Atualize o sistema de pacotes
    pacman -Syu

    # 2. Instale o pacote GTK3 (inclui a dependência de GObject)
    pacman -S mingw-w64-x86_64-gtk3
    ```

3. **Ajuste o PATH**: Adicione o diretório `bin` da sua instalação MSYS2 à variável de ambiente `PATH` do Windows.
    * Caminho padrão comum mas pode mudar um pouco(verificar o seu caminho): `C:\msys64\mingw64\bin`
4. **Reinicie o Terminal**: Feche e reabra o seu terminal ou VS Code para que as alterações do PATH sejam carregadas.

---

## ⚡ Inicialização e Acesso

### Execução do Servidor

Na raiz do projeto, execute o comando para iniciar o servidor LangGraph:

```bash
langgraph dev
```

### Rastreamento e Debug

O terminal irá gerar uma URL local (geralmente `http://localhost:8123` ou similar). Abra esta URL no seu navegador para acessar:

* **LangGraph Studio:** Interface visual para interagir e ver o estado do seu grafo.
* **LangSmith:** Logs detalhados e rastreamento de cada etapa da execução do agente (se as chaves estiverem configuradas).

---

## 🛠️ Tecnologias Utilizadas

* [LangGraph](https://langchain-ai.github.io/langgraph/) - Orquestração de agentes.
* [Qdrant](https://qdrant.tech/) - Banco de dados vetorial (Vector Database).
* [WeasyPrint](https://weasyprint.org/) - Renderização de HTML/CSS para PDF.
* [Google Gemini](https://ai.google.dev/) - Modelo de Linguagem (LLM).
