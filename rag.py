# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = "llama3.2:latest", embedding_model: str = "mxbai-embed-large"):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        # 带历史对话的 Prompt 模板
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.

            Chat history:
            {history}

            Context:
            {context}

            Question:
            {question}

            Answer concisely and accurately in three sentences or less.
            """
        )

        self.vector_store = None
        self.retriever = None
        self.history = []  # 存储多轮对话历史

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db",
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        返回：(回答, 检索到的原文片段列表)
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question.", []

        # 只保留最近 4 条历史（2 轮问答），避免 Prompt 过长
        history_text = "\n".join(self.history[-4:])

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
            "history": history_text,
        }

        chain = (
            RunnablePassthrough()
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        logger.info("Generating response using the LLM.")
        answer = chain.invoke(formatted_input)

        # 把本轮问答存入历史
        self.history.append(f"User: {query}")
        self.history.append(f"Assistant: {answer}")

        # 检索溯源：返回原文片段
        sources = [doc.page_content for doc in retrieved_docs]

        return answer, sources

    def clear(self):
        """
        Reset the vector store, retriever, and conversation history.
        """
        logger.info("Clearing vector store, retriever, and history.")
        self.vector_store = None
        self.retriever = None
        self.history = []
