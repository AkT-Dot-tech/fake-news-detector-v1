import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── Load model ────────────────────────────────────────────────────
model      = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# ── Text cleaning ─────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    if len(words) == 0:
        return "empty"

    return " ".join(words)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050a14 !important;
    font-family: 'Sora', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,180,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(0,80,200,0.12) 0%, transparent 60%),
        #050a14 !important;
    min-height: 100vh;
}

/* animated grid lines */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,180,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,255,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: transparent !important;
    display: none;
}

[data-testid="block-container"] {
    padding: 2.5rem 1.5rem 4rem !important;
    max-width: 780px !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,180,255,0.10);
    border: 1px solid rgba(0,180,255,0.30);
    color: #00b4ff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin-bottom: 0.8rem;
}
.hero-title span {
    background: linear-gradient(90deg, #00b4ff, #0066ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #6b7fa3;
    font-size: 1rem;
    font-weight: 300;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto 2.5rem;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-bottom: 2.8rem;
    flex-wrap: wrap;
}
.stat-item {
    text-align: center;
}
.stat-num {
    font-size: 1.6rem;
    font-weight: 700;
    color: #00b4ff;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: #4a5978;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.25rem;
}
.stat-divider {
    width: 1px;
    background: rgba(255,255,255,0.06);
    align-self: stretch;
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(0,180,255,0.20); }
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4a5978;
    margin-bottom: 1rem;
}

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.2rem !important;
    resize: vertical !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(0,180,255,0.50) !important;
    box-shadow: 0 0 0 3px rgba(0,180,255,0.08) !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #2d3a52 !important; }
[data-testid="stTextArea"] label { display: none !important; }

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #0066ff, #00b4ff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(0,102,255,0.30) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,102,255,0.45) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── Result boxes ── */
.result-box {
    border-radius: 16px;
    padding: 2rem 2rem 1.6rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeSlide 0.5s ease;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-real {
    background: rgba(0,200,100,0.07);
    border: 1px solid rgba(0,200,100,0.25);
}
.result-fake {
    background: rgba(255,60,60,0.07);
    border: 1px solid rgba(255,60,60,0.25);
}
.result-icon { font-size: 3rem; margin-bottom: 0.6rem; }
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-label-real { color: #00c864; }
.result-label-fake { color: #ff3c3c; }
.result-heading {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.result-heading-real { color: #00e070; }
.result-heading-fake { color: #ff5555; }
.result-sub { color: #6b7fa3; font-size: 0.88rem; }

/* ── Confidence bar ── */
.conf-wrap {
    margin-top: 1.6rem;
    padding: 1.4rem 1.6rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
}
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.conf-name { font-size: 0.80rem; color: #6b7fa3; font-weight: 500; }
.conf-pct  { font-size: 0.80rem; font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.conf-pct-real { color: #00c864; }
.conf-pct-fake { color: #ff5555; }
.bar-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.bar-fill-real {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #00c864, #00ff88);
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.bar-fill-fake {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ff3c3c, #ff7070);
    transition: width 1s cubic-bezier(.4,0,.2,1);
}

/* ── How it works ── */
.steps {
    display: flex;
    gap: 0;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}
.step {
    flex: 1;
    min-width: 130px;
    text-align: center;
    padding: 1rem 0.5rem;
    position: relative;
}
.step:not(:last-child)::after {
    content: "→";
    position: absolute;
    right: -8px;
    top: 50%;
    transform: translateY(-50%);
    color: #2d3a52;
    font-size: 1.1rem;
}
.step-icon {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
}
.step-name {
    font-size: 0.75rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.2rem;
}
.step-desc {
    font-size: 0.68rem;
    color: #4a5978;
    line-height: 1.4;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    color: #2d3a52;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}

/* hide streamlit chrome */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔍 NLP · Machine Learning · Real-time</div>
    <div class="hero-title">Fake News <span>Detector</span></div>
    <div class="hero-sub">
        Paste any news article below. Our NLP model analyses the text
        and predicts whether it's real or fake in seconds.
    </div>
</div>

<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-num">98.85%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
        <div class="stat-num">44,898</div>
        <div class="stat-label">Articles Trained</div>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
        <div class="stat-num">5,000</div>
        <div class="stat-label">TF-IDF Features</div>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
        <div class="stat-num">LR</div>
        <div class="stat-label">Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT CARD ────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📋 News Article Input</div>', unsafe_allow_html=True)
news_input = st.text_area("", height=220,
    placeholder="Paste the full news article or a few paragraphs here...")
st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("Analyse Article →", use_container_width=True)

# ── RESULT ────────────────────────────────────────────────────────
if predict_btn:
    if not news_input.strip():
        st.markdown("""
        <div class="card" style="text-align:center; color:#6b7fa3;">
            ⚠️ &nbsp; Please paste a news article before analysing.
        </div>""", unsafe_allow_html=True)
    else:
        cleaned    = clean_text(news_input)
        vec        = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba      = model.predict_proba(vec)[0]

        fake_pct = round(proba[0] * 100, 1)
        real_pct = round(proba[1] * 100, 1)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-box result-real">
                <div class="result-icon">✅</div>
                <div class="result-label result-label-real">Prediction Result</div>
                <div class="result-heading result-heading-real">Real News</div>
                <div class="result-sub">The model predicts this article is likely genuine.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-fake">
                <div class="result-icon">🚨</div>
                <div class="result-label result-label-fake">Prediction Result</div>
                <div class="result-heading result-heading-fake">Fake News</div>
                <div class="result-sub">The model predicts this article may be misinformation.</div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bars
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-row">
                <span class="conf-name">Real News Confidence</span>
                <span class="conf-pct conf-pct-real">{real_pct}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill-real" style="width:{real_pct}%"></div>
            </div>
            <div class="conf-row">
                <span class="conf-name">Fake News Confidence</span>
                <span class="conf-pct conf-pct-fake">{fake_pct}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill-fake" style="width:{fake_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── HOW IT WORKS ─────────────────────────────────────────────────
st.markdown("""
<div class="card" style="margin-top:2rem;">
    <div class="card-title">⚙️ How It Works</div>
    <div class="steps">
        <div class="step">
            <div class="step-icon">📥</div>
            <div class="step-name">Input</div>
            <div class="step-desc">Raw news article text</div>
        </div>
        <div class="step">
            <div class="step-icon">🧹</div>
            <div class="step-name">Clean</div>
            <div class="step-desc">Remove noise &amp; stopwords</div>
        </div>
        <div class="step">
            <div class="step-icon">🔢</div>
            <div class="step-name">TF-IDF</div>
            <div class="step-desc">Text → numbers</div>
        </div>
        <div class="step">
            <div class="step-icon">🧠</div>
            <div class="step-name">Model</div>
            <div class="step-desc">Logistic Regression</div>
        </div>
        <div class="step">
            <div class="step-icon">📊</div>
            <div class="step-name">Result</div>
            <div class="step-desc">Fake or Real + confidence</div>
        </div>
    </div>
</div>

<div class="footer">
    Built with Python · NLTK · Scikit-learn · Streamlit &nbsp;|&nbsp; B.Tech CSE Project · IILM University
</div>
""", unsafe_allow_html=True)