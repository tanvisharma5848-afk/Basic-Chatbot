import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GroqChat",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #080b12;
    --surface:   #0e1320;
    --border:    #1e2740;
    --accent:    #00e5ff;
    --accent2:   #7c3aed;
    --text:      #e8eaf6;
    --muted:     #5c6585;
    --user-bg:   #0d1f3c;
    --ai-bg:     #0c1a1e;
    --radius:    14px;
}

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 760px !important; margin: 0 auto; }

.hero {
    text-align: center;
    padding: 48px 24px 28px;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 320px; height: 120px;
    background: radial-gradient(ellipse at center, rgba(0,229,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid rgba(0,229,255,0.25);
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
    background: rgba(0,229,255,0.05);
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 48px !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px;
    color: #ffffff !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.1;
}
.hero h1 span {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 13px;
    color: var(--muted);
    margin-top: 10px;
    letter-spacing: 0.5px;
}
.status-row {
    display: flex;
    justify-content: center;
    margin: 4px 0 20px;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 6px 16px;
    border-radius: 20px;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 0 24px 24px;
}
.chat-wrap {
    padding: 0 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-height: 200px;
}
.msg-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.msg-row.user { flex-direction: row-reverse; }
.avatar {
    width: 34px; height: 34px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}
.avatar.user-av {
    background: linear-gradient(135deg, #1e3a5f, #0d1f3c);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent);
}
.avatar.ai-av {
    background: linear-gradient(135deg, #0c1a1e, #071013);
    border: 1px solid rgba(124,58,237,0.4);
    color: var(--accent2);
}
.bubble {
    max-width: 78%;
    padding: 13px 17px;
    border-radius: var(--radius);
    font-size: 13.5px;
    line-height: 1.65;
    position: relative;
    white-space: pre-wrap;
    word-break: break-word;
}
.bubble.user-bubble {
    background: var(--user-bg);
    border: 1px solid rgba(0,229,255,0.15);
    border-top-right-radius: 4px;
    color: #c8deff;
}
.bubble.ai-bubble {
    background: var(--ai-bg);
    border: 1px solid rgba(124,58,237,0.15);
    border-top-left-radius: 4px;
    color: var(--text);
}
.bubble-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.user-bubble .bubble-label { color: rgba(0,229,255,0.5); }
.ai-bubble .bubble-label { color: rgba(124,58,237,0.6); }

.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: var(--muted);
}
.empty-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    max-width: 420px;
    margin: 24px auto 0;
}
.suggestion-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 12px;
    text-align: left;
    line-height: 1.5;
}
.chip-icon { font-size: 18px; margin-bottom: 5px; display: block; }
.chip-label { color: var(--text); display: block; font-weight: 500; }
.chip-sub { color: var(--muted); font-size: 11px; }

.input-section {
    padding: 20px 20px 28px;
    margin-top: 16px;
    position: relative;
}
.input-section::before {
    content: '';
    position: absolute;
    top: 0; left: 20px; right: 20px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
}

.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13.5px !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s !important;
    caret-color: var(--accent) !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(0,229,255,0.45) !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,0.06) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--muted) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #5b21b6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 12px 26px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 26px rgba(124,58,237,0.45) !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.chat-footer {
    text-align: center;
    font-size: 11px;
    color: var(--muted);
    padding: 0 0 20px;
    letter-spacing: 0.5px;
}
.chat-footer span { color: rgba(0,229,255,0.4); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ─── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=2,
    )

model = get_model()

# ─── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">⚡ Powered by Groq</div>
    <h1>Groq<span>Chat</span></h1>
    <p class="hero-sub">LLaMA 3.1 · 8B · Instant Inference</p>
</div>
<div class="status-row">
    <div class="status-badge">
        <div class="status-dot"></div>
        <span>llama-3.1-8b-instant · Online</span>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── Chat History ────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:36px;margin-bottom:8px;">⚡</div>
        <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:#e8eaf6;margin-bottom:6px;">
            Start a conversation
        </div>
        <div style="font-size:12px;letter-spacing:0.5px;">Ask anything. Groq responds at lightning speed.</div>
        <div class="empty-grid">
            <div class="suggestion-chip">
                <span class="chip-icon">🧠</span>
                <span class="chip-label">Explain a concept</span>
                <span class="chip-sub">Break down complex ideas simply</span>
            </div>
            <div class="suggestion-chip">
                <span class="chip-icon">💻</span>
                <span class="chip-label">Write some code</span>
                <span class="chip-sub">Generate or debug any snippet</span>
            </div>
            <div class="suggestion-chip">
                <span class="chip-icon">✍️</span>
                <span class="chip-label">Draft content</span>
                <span class="chip-sub">Emails, posts, documents</span>
            </div>
            <div class="suggestion-chip">
                <span class="chip-icon">🔍</span>
                <span class="chip-label">Analyze & summarize</span>
                <span class="chip-sub">Process any text or data</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar user-av">U</div>
                <div class="bubble user-bubble">
                    <div class="bubble-label">You</div>
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row">
                <div class="avatar ai-av">G</div>
                <div class="bubble ai-bubble">
                    <div class="bubble-label">GroqChat</div>
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── Input Area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="input-section">', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        label="",
        placeholder="Type a message…",
        key="user_input",
        label_visibility="collapsed"
    )
with col2:
    send_clicked = st.button("Send ⚡", key="send_btn")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="chat-footer">
    Built with <span>LangChain + Groq</span> · LLaMA 3.1 8B Instant
</div>
""", unsafe_allow_html=True)

# ─── Logic: Capture Input → Store as Pending → Clear Field via rerun ─────────────
if send_clicked and user_input.strip():
    st.session_state.pending_prompt = user_input.strip()
    st.rerun()

# ─── Logic: Process Pending Prompt (field is now cleared) ────────────────────────
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None  # clear immediately to prevent re-processing

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner(""):
        try:
            response = model.invoke(prompt)
            ai_reply = response.content
        except Exception as e:
            ai_reply = f"⚠️ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
    st.rerun()