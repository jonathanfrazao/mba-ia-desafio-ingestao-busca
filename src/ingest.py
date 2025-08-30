import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _require_env() -> None:
    for k in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if not os.getenv(k):
            raise RuntimeError(f"A variável de ambiente {k} não está definida.")


def _resolve_pdf_path() -> Path:
    pdf_env = os.getenv("PDF_PATH")
    if not pdf_env:
        raise RuntimeError("A variável de ambiente PDF_PATH não está definida.")

    pdf_path = Path(pdf_env)
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF não encontrado em: {pdf_path}. "
            f"Defina corretamente o PDF_PATH no .env"
        )
    return pdf_path


def _split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False,
    )
    splits = splitter.split_documents(docs)

    if not splits:
        print("[INGEST] Split não gerou nenhum chunk.")
        return []

    print(f"[INGEST] Chunks gerados: {len(splits)} "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return splits


def _prepare_documents(splits):
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in (d.metadata or {}).items() if v not in ("", None)},
        )
        for d in splits
    ]
    return enriched


def ingest_pdf() -> None:
    # 1) Verificações iniciais
    _require_env()

    # 2) Resolver caminho do PDF
    pdf_path = _resolve_pdf_path()
    print(f"[INGEST] Lendo PDF: {pdf_path}")

    # 3) Carregar PDF
    docs = PyPDFLoader(str(pdf_path)).load()
    if not docs:
        print("[INGEST] Nenhuma página carregada do PDF.")
        return
    print(f"[INGEST] Páginas carregadas: {len(docs)}")

    # 4) Split
    splits = _split_documents(docs)

    # Divide os documentos em chunks (CHUNK_SIZE/CHUNK_OVERLAP) e retorna os splits (ou [] se nenhum)
    enriched = _prepare_documents(splits)

    # 6) Define 'source' com o caminho do PDF para rastreabilidade e limpeza seletiva no banco
    for d in enriched:
        d.metadata["source"] = str(pdf_path)

    # 7) Ids determinísticos
    ids = [f"{pdf_path.stem}-chunk-{i}" for i in range(len(enriched))]

    # 8) Embeddings
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    print(f"[INGEST] Embedding model: {embeddings.model}")

    # 9) Vetor store (pgVector)
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    # 10) Deleta tudo desse PDF antes de reingestar
    try:
        store.delete(filter={"source": str(pdf_path)})
    except TypeError:
        store.delete(ids=ids)

    # 11) Gravar
    store.add_documents(documents=enriched, ids=ids)
    print(f"[INGEST] Ingestão concluída! {len(enriched)} chunks inseridos na coleção '{os.getenv('PG_VECTOR_COLLECTION_NAME')}'.")
    print("[INGEST] Pronto para executar a busca (search/chat).")


if __name__ == "__main__":
    ingest_pdf()
