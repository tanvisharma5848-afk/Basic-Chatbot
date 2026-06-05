import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import time

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #111118;
    --border:  #1e1e2e;
    --accent:  #7c6fff;
    --accent2: #ff6f91;
    --text:    #e8e8f0;
    --muted:   #5a5a72;
    --radius:  14px;
}

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: var(--bg);
    color: var(--text);
}

.main { background: var(--bg); }
.block-container { padding: 1.5rem 2rem 5rem 2rem; max-width: 860px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }

/* Header */
.rag-header { text-align: center; padding: 2rem 0 1rem 0; }
.rag-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #7c6fff 0%, #ff6f91 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; line-height: 1.1;
}
.rag-subtitle {
    font-size: 0.72rem; color: var(--muted);
    letter-spacing: 3px; text-transform: uppercase; margin-top: 0.4rem;
}
.styled-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 1.2rem 0; opacity: 0.35;
}

/* Row wrapper for each message */
.msg-row {
    display: flex; gap: 10px; margin: 0.9rem 0; align-items: flex-start;
}
.msg-avatar {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; flex-shrink: 0;
    font-family: 'Syne', sans-serif; font-weight: 700; color: #fff;
}
.avatar-user { background: linear-gradient(135deg, #7c6fff, #5a4de0); }
.avatar-ai   { background: linear-gradient(135deg, #ff6f91, #c0415e); }

/* Bubble shell — text rendered by Streamlit INSIDE this via container trick */
.bubble-shell {
    flex: 1; border-radius: var(--radius);
    padding: 0.8rem 1rem 0.6rem 1rem;
    font-size: 0.87rem; line-height: 1.7; word-wrap: break-word;
}
.bubble-user {
    background: #16162a;
    border: 1px solid rgba(124,111,255,0.22);
}
.bubble-ai {
    background: #0f1624;
    border: 1px solid rgba(255,111,145,0.16);
}

/* Meta row under AI answer */
.meta-bar {
    display: flex; flex-wrap: wrap; gap: 8px;
    margin-top: 8px; align-items: center;
}
.meta-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 2px 10px; font-size: 0.68rem; color: var(--muted);
}
.meta-chip b { color: var(--accent); }
.chunk-pill {
    background: rgba(124,111,255,0.07);
    border: 1px solid rgba(124,111,255,0.28);
    color: #9d8fff; font-size: 0.67rem;
    padding: 2px 9px; border-radius: 99px;
}

/* Input */
.stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.87rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,111,255,0.12) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #7c6fff, #5a4de0) !important;
    color: #fff !important; border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; font-size: 0.87rem !important;
    padding: 0.55rem 1.2rem !important; width: 100% !important;
}
.stButton button:hover { opacity: 0.85 !important; }

.clear-btn button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important; font-size: 0.75rem !important;
}
.clear-btn button:hover {
    border-color: var(--accent2) !important;
    color: var(--accent2) !important;
}

/* Sidebar stats */
.sidebar-stat {
    background: rgba(124,111,255,0.05);
    border: 1px solid rgba(124,111,255,0.14);
    border-radius: 10px; padding: 0.65rem 1rem;
    margin-bottom: 0.55rem; font-size: 0.76rem; color: var(--muted);
}
.sidebar-stat strong {
    color: var(--accent); font-size: 1.05rem;
    display: block; font-family: 'Syne', sans-serif; font-weight: 700;
}

/* Expander */
.stExpander { border: 1px solid var(--border) !important; border-radius: 10px !important; }
details > summary { font-size: 0.74rem !important; color: var(--muted) !important; }

/* Welcome */
.welcome-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 2rem; text-align: center; margin: 1.5rem 0;
}
.welcome-card h3 {
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 1.15rem; color: var(--text); margin: 0.5rem 0;
}
.welcome-card p { font-size: 0.80rem; color: var(--muted); margin: 0; line-height: 1.6; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

/* Streamlit markdown text inside bubbles */
.bubble-shell p { margin: 0; }
.bubble-shell p + p { margin-top: 0.5rem; }

/* Hide chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Selectbox / slider text */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.80rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Cached Resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource(show_spinner=False)
def load_vector_store(_embeddings):
    return Chroma(persist_directory="./chroma_langchain_db", embedding_function=_embeddings)

@st.cache_resource(show_spinner=False)
def load_model(model_name: str, temperature: float):
    return ChatGroq(model=model_name, temperature=temperature, max_retries=2)


# ─── Session State ────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "total_queries" not in st.session_state: st.session_state.total_queries = 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.4rem 0;'>
      <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1rem;
                  background:linear-gradient(135deg,#7c6fff,#ff6f91);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;'>⚙ CONFIGURATION</div>
    </div>
    <hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,#1e1e2e,transparent);margin:0.6rem 0;'>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    top_k       = st.slider("Retrieved Chunks (k)", 1, 8, 3)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;color:#5a5a72;margin-bottom:0.6rem;'>Session Stats</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='sidebar-stat'><strong>{st.session_state.total_queries}</strong>Total Queries</div>
    <div class='sidebar-stat'><strong>{len(st.session_state.messages)}</strong>Messages</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;color:#5a5a72;margin-bottom:0.6rem;'>Vector Store</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(255,255,255,0.02);border:1px solid #1e1e2e;border-radius:10px;
                padding:0.75rem 1rem;font-size:0.73rem;color:#5a5a72;line-height:1.8;'>
        📁 <span style='color:#7c6fff'>./chroma_langchain_db</span><br>
        🤗 all-mpnet-base-v2<br>
        ⚡ Groq Inference
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑 Clear Chat History"):
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Load Resources ───────────────────────────────────────────────────────────
with st.spinner("Initialising models..."):
    embeddings    = load_embeddings()
    vector_store  = load_vector_store(embeddings)
    model         = load_model(model_choice, temperature)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='rag-header'>
  <div class='rag-title'>🔮 RAG Assistant</div>
  <div class='rag-subtitle'>Retrieval-Augmented Generation · Powered by Groq</div>
</div>
<div class='styled-divider'></div>
""", unsafe_allow_html=True)


# ─── Chat Display ─────────────────────────────────────────────────────────────
def render_message(msg: dict):
    role = msg["role"]

    if role == "user":
        # Avatar + bubble shell (layout only, no dynamic text inside HTML)
        col_av, col_bub = st.columns([0.06, 0.94])
        with col_av:
            st.markdown("<div class='msg-avatar avatar-user'>U</div>", unsafe_allow_html=True)
        with col_bub:
            st.markdown("<div class='bubble-shell bubble-user'>", unsafe_allow_html=True)
            st.markdown(msg["content"])          # ← plain Streamlit, no HTML injection
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        col_av, col_bub = st.columns([0.06, 0.94])
        with col_av:
            st.markdown("<div class='msg-avatar avatar-ai'>AI</div>", unsafe_allow_html=True)
        with col_bub:
            st.markdown("<div class='bubble-shell bubble-ai'>", unsafe_allow_html=True)
            st.markdown(msg["content"])          # ← plain Streamlit, no HTML injection

            # Meta bar (static values only — safe to inline)
            if msg.get("elapsed") is not None:
                pills_html = "".join(
                    f"<span class='chunk-pill'>📄 Chunk {i+1}</span>"
                    for i in range(msg.get("chunks", 0))
                )
                st.markdown(f"""
                <div class='meta-bar'>
                    <div class='meta-chip'>⏱ <b>{msg['elapsed']:.2f}s</b></div>
                    <div class='meta-chip'>📦 <b>{msg['chunks']}</b> chunks</div>
                    {pills_html}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Collapsible context viewer
            if msg.get("sources"):
                with st.expander(f"🔍 View Retrieved Context ({len(msg['sources'])} chunks)"):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"<div style='font-size:0.65rem;text-transform:uppercase;"
                            f"letter-spacing:2px;color:#5a5a72;margin-bottom:4px;'>Chunk {i+1}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.02);border:1px solid #1e1e2e;"
                            f"border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.5rem;"
                            f"font-size:0.76rem;color:#8888aa;line-height:1.6;'>{src}</div>",
                            unsafe_allow_html=True,
                        )


if not st.session_state.messages:
    st.markdown("""
    <div class='welcome-card'>
        <div style='font-size:2.2rem;'>💡</div>
        <h3>Ask me anything from your knowledge base</h3>
        <p>I retrieve the most relevant context from your vector store<br>
        and generate a precise, grounded answer using Groq LLMs.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render_message(msg)


st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)


# ─── Input ────────────────────────────────────────────────────────────────────
st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

col_inp, col_btn = st.columns([5, 1])
with col_inp:
    user_input = st.text_input(
        label="query",
        placeholder="Ask a question from your knowledge base...",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send_clicked = st.button("Send →", use_container_width=True)


# ─── Query + Response Logic ───────────────────────────────────────────────────
if send_clicked and user_input.strip():
    query = user_input.strip()

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.total_queries += 1

    with st.spinner("Searching vector store & generating answer..."):
        t0      = time.time()
        results = vector_store.similarity_search(query, k=top_k)
        chunks  = [r.page_content for r in results]
        context = "\n\n".join(chunks)

        prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find relevant information in the knowledge base."

Context:
{context}

Question: {query}

Answer:"""

        response = model.invoke(prompt)
        elapsed  = time.time() - t0

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response.content,
        "sources": chunks,
        "elapsed": elapsed,
        "chunks":  len(chunks),
    })

    st.rerun()