# Desafio MBA Engenharia de Software com IA - Full Cycle

## Ingestão e Busca Semântica (LangChain + Postgres/pgVector)

Aplicação que ingere um PDF no PostgreSQL (pgVector) e responde perguntas via CLI usando somente o conteúdo do PDF.

## 📋 Pré-requisitos

* Python 3.13+
* Docker + Docker Compose
* Chave da OpenAI

## ⚙️ Configuração

### 1. Configurar `.env` (na raiz do projeto)

Não use aspas nos valores.

```env
OPENAI_API_KEY=coloque_sua_api_key_aqui
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag?connect_timeout=10
PG_VECTOR_COLLECTION_NAME=gpt5_collection
PDF_PATH=./document.pdf
CHAT_MODEL=gpt-5-nano
```

Nota: Use o mesmo `OPENAI_EMBEDDING_MODEL` na ingestão e na busca.

### 2. Subir o banco (pgVector)

Na raiz do projeto:

```bash
docker compose up -d
```

### 3. Configurar ambiente Python

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## 🚀 Uso

### Ingestão do PDF

```bash
python src/ingest.py
```

Saída final esperada (exemplo):

```text
[INGEST] Pronto para executar a busca (search/chat).
```

### Chat (CLI)

```bash
python src/chat.py
```

Exemplo de uso:

```text
PERGUNTA: qual o faturamento registrado da empresa Dourado Petróleo EPP?
RESPOSTA: R$ 105.629,26.
```

Caso de pergunta fora do contexto:

```text
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

