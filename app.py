import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import SecretStr

load_dotenv()

st.set_page_config(page_title="Asistente RAG · PDFs", page_icon="📄", layout="wide")
st.title("📄 Chat Asistente para documentos PDF")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),          
    ("chat_history", []),     
    ("rag_chain", None),
    ("processed_file", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Chain builder ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_chain(file_bytes: bytes, file_name: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    api_key_raw = os.getenv("ANTHROPIC_API_KEY")

    if not api_key_raw:
        raise ValueError("¡La ANTHROPIC_API_KEY no está configurada en el entorno!")

    llm = ChatAnthropic(
        model_name="claude-sonnet-4-6",
        api_key=SecretStr(api_key_raw),
        temperature=0,
        timeout=None,
        stop=None
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Dado el historial de conversación y la última pregunta del usuario, "
         "reformula la pregunta para que sea independiente y comprensible sin "
         "el historial. Si ya es clara, devuélvela tal cual."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Eres un asistente que responde preguntas basándose exclusivamente en "
         "los documentos proporcionados. Usa los fragmentos de contexto para "
         "elaborar una respuesta clara y concisa. Si la respuesta no está en "
         "el contexto, dilo explícitamente.\n\nContexto:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain, len(chunks)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("No se detectó ANTHROPIC_API_KEY en el entorno.")

    st.divider()
    st.subheader("📁 Documento")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
    process_btn = st.button("🚀 Procesar documento", disabled=uploaded_file is None)

    if process_btn and uploaded_file:
        file_bytes = uploaded_file.read()
        with st.spinner("Procesando PDF… (puede tardar unos segundos la primera vez)"):
            try:
                chain, n_chunks = build_rag_chain(file_bytes, uploaded_file.name)
                st.session_state.rag_chain = chain
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.processed_file = uploaded_file.name
                st.success(f"✅ Listo — {n_chunks} fragmentos indexados.")
            except Exception as exc:
                st.error(f"Error al procesar: {exc}")

    if st.session_state.processed_file:
        st.info(f"Documento activo: **{st.session_state.processed_file}**")

    st.divider()
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Embeddings: all-MiniLM-L6-v2 (local) · LLM: Claude 3.5 Sonnet")


# ── Chat area ─────────────────────────────────────────────────────────────────
if not st.session_state.rag_chain:
    st.info("👈 Sube un PDF en el panel lateral y pulsa **Procesar documento** para empezar.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Fuentes utilizadas"):
                for i, doc in enumerate(msg["sources"], 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Fragmento {i}** · Página {page + 1}")
                    st.caption(
                        doc.page_content[:300]
                        + ("…" if len(doc.page_content) > 300 else "")
                    )

if question := st.chat_input("Haz una pregunta sobre el documento…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando respuesta…"):
            try:
                result = st.session_state.rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history,
                })
                answer = result["answer"]
                sources = result.get("context", [])

                st.markdown(answer)
                if sources:
                    with st.expander("📎 Fuentes utilizadas"):
                        for i, doc in enumerate(sources, 1):
                            page = doc.metadata.get("page", "?")
                            st.markdown(f"**Fragmento {i}** · Página {page + 1}")
                            st.caption(
                                doc.page_content[:300]
                                + ("…" if len(doc.page_content) > 300 else "")
                            )

                st.session_state.chat_history.extend([
                    HumanMessage(content=question),
                    AIMessage(content=answer),
                ])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                err = f"Error al generar respuesta: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
