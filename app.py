"""
College Information Assistant — Premium Portfolio Edition
A production-grade Semantic RAG application built with Streamlit, Ollama, and nomic-embed-text.

Architecture:
- Smart Document & Intent Routing (Prioritizes Master Calendars & Curriculum tables)
- Typo-Tolerant Query Normalizer (Damerau-Levenshtein fuzzy matching)
- Dense 768-dim Vector Search (nomic-embed-text normalized cosine similarity)
- Conversational Academic Advisor Persona (Direct, structured, date-accurate responses)
- Modern Glassmorphism Streamlit UI with live confidence metrics
"""

import hashlib
import json
import os
import re
import time

import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_FOLDER = os.path.join(BASE_DIR, "documents")
CACHE_FOLDER = os.path.join(BASE_DIR, ".rag_cache")

LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.38
NUM_CTX = 4096
HISTORY_TURNS = 4

MONTHS_MAP = {
    "Jun-25": ("June", "2025"),
    "Jul-25": ("July", "2025"),
    "Aug-25": ("August", "2025"),
    "Sep-25": ("September", "2025"),
    "Oct-25": ("October", "2025"),
    "Nov-25": ("November", "2025"),
    "Dec-25": ("December", "2025"),
    "Jan-26": ("January", "2026"),
    "Feb-26": ("February", "2026"),
    "Mar-26": ("March", "2026"),
    "Apr-26": ("April", "2026"),
    "May-26": ("May", "2026"),
    "Jun-26": ("June", "2026"),
    "Jul-26": ("July", "2026"),
}

BASE_VOCABULARY = {
    "academic", "calendar", "semester", "semesters", "examination", "examinations",
    "exam", "exams", "syllabus", "curriculum", "mechanical", "engineering", "evaluation",
    "assessment", "commencement", "credits", "credit", "prerequisite", "prerequisites",
    "holiday", "holidays", "schedule", "internal", "grade", "grades", "grading",
    "marks", "btech", "mtech", "attendance", "laboratory", "course", "courses",
    "project", "regulations", "elective", "electives", "faculty", "admission", "department",
    "automation", "robotics", "mechanics", "thermodynamics", "physics", "maths"
}

# ==========================================
# FUZZY SPELL CORRECTION
# ==========================================

def edit_distance(s1, s2):
    d = {}
    len1, len2 = len(s1), len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,
                d[(i, j - 1)] + 1,
                d[(i - 1, j - 1)] + cost
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)

    return d[(len1 - 1, len2 - 1)]


def normalize_query(query, vocabulary):
    tokens = re.findall(r"[a-zA-Z0-9]+|[^\w\s]", query)
    corrected_tokens = []
    modified = False

    for tok in tokens:
        tok_lower = tok.lower()
        if not tok.isalnum() or len(tok) <= 3 or tok_lower in vocabulary:
            corrected_tokens.append(tok)
            continue

        best_match = None
        min_dist = 99
        for term in vocabulary:
            max_allowed = 2 if len(tok_lower) > 4 else 1
            dist = edit_distance(tok_lower, term)
            if dist <= max_allowed and dist < min_dist:
                min_dist = dist
                best_match = term

        if best_match:
            corrected_tokens.append(best_match)
            modified = True
        else:
            corrected_tokens.append(tok)

    corrected_str = ""
    for i, tok in enumerate(corrected_tokens):
        if i > 0 and tok.isalnum() and corrected_tokens[i - 1].isalnum():
            corrected_str += " " + tok
        elif i > 0 and tok.isalnum() and corrected_tokens[i - 1] in {"?", "!", ".", ","}:
            corrected_str += " " + tok
        else:
            corrected_str += tok

    return corrected_str.strip(), modified


# ==========================================
# STREAMLIT SAFE HELPERS
# ==========================================

def safe_update_status(status_obj, label=None, state=None, expanded=None):
    if status_obj is None:
        return
    try:
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        if state is not None:
            kwargs["state"] = state
        if expanded is not None:
            kwargs["expanded"] = expanded
        if hasattr(status_obj, "update"):
            status_obj.update(**kwargs)
    except Exception:
        pass


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60}m {seconds % 60}s" if seconds >= 60 else f"{seconds}s"


GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|hii+|good\s*(morning|afternoon|evening)|thanks|thank\s*you|ok|okay)\W*$",
    re.I,
)

# ==========================================
# STRUCTURED PARSING & CHUNKING
# ==========================================

def list_pdfs():
    os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
    return sorted(
        f for f in os.listdir(DOCUMENT_FOLDER) if f.lower().endswith(".pdf")
    )


def parse_academic_calendar(reader, filename):
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        m_name, year = "", ""
        for m_code, (m, y) in MONTHS_MAP.items():
            if m_code in text:
                m_name, year = m, y
                break

        if not m_name:
            m_name, year = f"Page {page_no}", "2025-2026"

        events = []
        for line in text.split("\n"):
            line = line.strip()
            m = re.match(
                r"^(\d{1,2})\s+([A-Za-z]{3})\s+.*?(Commencement.*?|Mid Semester.*?|End Semester.*?|Holiday.*?|Teachers Camp.*?|Vacation.*?|Re-opening.*?|Examinat.*?|Amma.*?)$",
                line,
                re.I,
            )
            if m:
                day_num, day_name, event = m.groups()
                events.append(f"• {m_name} {day_num}, {year} ({day_name}): {event.strip()}")
            elif any(
                k in line.lower()
                for k in ["commencement", "exam", "holiday", "jayanthi", "pongal", "diwali"]
            ):
                events.append(f"• {line}")

        chunk_text = (
            f"[Document: Academic Calendar AY 2025–2026 | Month: {m_name} {year} (Page {page_no})]\n"
            + "\n".join(events)
        )
        chunks.append(
            {
                "text": chunk_text,
                "source": filename,
                "page_start": page_no,
                "page_end": page_no,
            }
        )
    return chunks


def parse_curriculum_and_syllabus(reader, filename):
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        section_title = f"Page {page_no}"
        first_lines = [l.strip() for l in text.split("\n") if l.strip()][:4]
        for l in first_lines:
            if any(k in l.lower() for k in ["semester", "evaluation", "specialization", "credit structure", "electives"]):
                section_title = l
                break

        # Curriculum master tables (pages 1 to 20): keep page whole
        if page_no <= 20:
            chunk_text = f"[Document: B.Tech Mechanical Engineering Curriculum | Section: {section_title} (Page {page_no})]\n{text}"
            chunks.append(
                {
                    "text": chunk_text,
                    "source": filename,
                    "page_start": page_no,
                    "page_end": page_no,
                }
            )
        else:
            words = text.split()
            if len(words) <= 350:
                chunk_text = f"[Document: Course Syllabus | Section: {section_title} (Page {page_no})]\n{text}"
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": filename,
                        "page_start": page_no,
                        "page_end": page_no,
                    }
                )
            else:
                step = 250
                for start in range(0, len(words), step):
                    sub_text = " ".join(words[start : start + 300])
                    chunk_text = f"[Document: Course Syllabus | Section: {section_title} (Page {page_no})]\n{sub_text}"
                    chunks.append(
                        {
                            "text": chunk_text,
                            "source": filename,
                            "page_start": page_no,
                            "page_end": page_no,
                        }
                    )
    return chunks


def create_document_chunks(filepath, filename):
    reader = PdfReader(filepath)
    fname_lower = filename.lower()
    if "calendar" in fname_lower:
        return parse_academic_calendar(reader, filename)
    else:
        return parse_curriculum_and_syllabus(reader, filename)


def page_label(chunk):
    if chunk["page_start"] == chunk["page_end"]:
        return f"page {chunk['page_start']}"
    return f"pages {chunk['page_start']}–{chunk['page_end']}"


# ==========================================
# OLLAMA & VECTOR EMBEDDINGS
# ==========================================

def check_ollama():
    try:
        response = ollama.list()
        models = response["models"] if isinstance(response, dict) else response.models
        names = set()
        for m in models:
            name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
            if name:
                names.add(name)
                names.add(name.split(":")[0])
    except Exception as e:
        st.error("⚠️ Unable to connect to Ollama.")
        st.markdown("Ensure Ollama is running (`ollama serve`), then refresh this page.")
        st.caption(f"Details: {e}")
        st.stop()

    missing = [
        m for m in (LLM_MODEL, EMBEDDING_MODEL)
        if m not in names and m.split(":")[0] not in names
    ]
    if missing:
        st.error("⚠️ Required models missing from Ollama:")
        st.code("\n".join(f"ollama pull {m}" for m in missing), language="bash")
        st.stop()


def embed_texts(texts, kind="doc"):
    prefix = "search_document: " if kind == "doc" else "search_query: "
    response = ollama.embed(
        model=EMBEDDING_MODEL, input=[prefix + t for t in texts]
    )
    vectors = np.array(response["embeddings"], dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


# ==========================================
# KNOWLEDGE BASE & CACHING
# ==========================================

def fingerprint(files):
    h = hashlib.sha256()
    h.update(b"v4_boosted_routed_rag")
    for f in files:
        p = os.path.join(DOCUMENT_FOLDER, f)
        st_ = os.stat(p)
        h.update(f"{f}|{st_.st_size}|{int(st_.st_mtime)}".encode())
    h.update(f"{EMBEDDING_MODEL}".encode())
    return h.hexdigest()


def build_knowledge_base(force=False):
    files = list_pdfs()
    if not files:
        return None

    os.makedirs(CACHE_FOLDER, exist_ok=True)
    fp = fingerprint(files)
    meta_path = os.path.join(CACHE_FOLDER, "meta.json")
    chunks_path = os.path.join(CACHE_FOLDER, "chunks.json")
    emb_path = os.path.join(CACHE_FOLDER, "embeddings.npy")

    started = time.time()

    if not force and all(os.path.exists(p) for p in (meta_path, chunks_path, emb_path)):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") == fp:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                embeddings = np.load(emb_path)
                vocab = set(meta.get("vocab", [])) | BASE_VOCABULARY
                return {
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "files": files,
                    "vocab": vocab,
                    "skipped": meta.get("skipped", []),
                }
        except Exception:
            pass

    with st.status("Structuring knowledge base...", expanded=True) as status:
        st.write(f"📄 **Step 1/2** — Structuring {len(files)} PDF document(s)...")
        read_bar = st.progress(0.0)
        chunks, skipped = [], []
        dynamic_vocab = set(BASE_VOCABULARY)

        for i, name in enumerate(files):
            read_bar.progress(i / len(files), text=f"Processing {name}")
            try:
                file_path = os.path.join(DOCUMENT_FOLDER, name)
                doc_chunks = create_document_chunks(file_path, name)
                chunks.extend(doc_chunks)
                st.write(f"   • **{name}**: extracted {len(doc_chunks)} structured passages")
            except Exception as e:
                skipped.append(f"{name} ({e})")
                continue

        read_bar.progress(1.0, text="PDFs structured successfully.")

        if not chunks:
            safe_update_status(status, label="No readable text found.", state="error")
            return {"chunks": [], "embeddings": None, "files": files, "vocab": BASE_VOCABULARY, "skipped": skipped}

        st.write(f"🧠 **Step 2/2** — Creating semantic embeddings ({len(chunks)} passages)...")
        emb_bar = st.progress(0.0)
        vectors = []
        t0 = time.time()
        batch_size = 32

        for start in range(0, len(chunks), batch_size):
            batch = [c["text"] for c in chunks[start : start + batch_size]]
            vectors.append(embed_texts(batch, "doc"))
            done = min(start + batch_size, len(chunks))
            elapsed = time.time() - t0
            eta = (elapsed / done) * (len(chunks) - done) if done > 0 else 0
            emb_bar.progress(done / len(chunks), text=f"Embedded {done}/{len(chunks)} ({fmt_time(eta)} left)")

        embeddings = np.vstack(vectors)

        np.save(emb_path, embeddings)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fp, "skipped": skipped, "vocab": list(dynamic_vocab)[:500]}, f)

        safe_update_status(
            status,
            label=f"Ready! Indexed {len(chunks)} passages from {len(files)} document(s) in {fmt_time(time.time() - started)}.",
            state="complete",
            expanded=False,
        )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "files": files,
        "vocab": dynamic_vocab,
        "skipped": skipped,
    }


# ==========================================
# SMART INTENT-ROUTED RETRIEVAL
# ==========================================

def retrieve_semantic_context(query, kb, top_k=DEFAULT_TOP_K, min_sim=SIMILARITY_THRESHOLD):
    """
    Semantic retrieval with intent routing:
    - Normalizes typos in user query.
    - Accurately routes dates to Calendar and curriculum to Master Tables (p. 6–20).
    - Prevents textbook references from drowning out core answers.
    """
    if kb is None or kb["embeddings"] is None or len(kb["chunks"]) == 0:
        return [], query, False

    vocab = kb.get("vocab", BASE_VOCABULARY)
    clean_query, corrected = normalize_query(query, vocab)
    q_lower = clean_query.lower()

    # Fast vector embedding
    q_vec = embed_texts([clean_query], kind="query")
    sims = (kb["embeddings"] @ q_vec.T).squeeze()

    if sims.ndim == 0:
        sims = np.array([float(sims)])

    # Intent flags
    is_calendar_query = any(
        k in q_lower
        for k in ["date", "dates", "calendar", "when", "exam", "exams", "commence", "commencement", "holiday", "holidays", "schedule", "re-open", "vacation"]
    )
    is_curriculum_overview = any(
        k in q_lower
        for k in ["semester", "sem", "subject", "subjects", "course", "courses", "credit", "credits", "evaluation", "grading", "internal", "external", "specialization", "specializations", "electives"]
    )

    boosted_sims = sims.copy()
    for idx, c in enumerate(kb["chunks"]):
        src = c["source"].lower()
        pstart = c.get("page_start", 999)

        # Prioritize Academic Calendar for date/event queries
        if is_calendar_query and "calendar" in src:
            boosted_sims[idx] += 0.22

        # Prioritize Curriculum Master Pages (pages 6 to 20) over individual textbook pages
        if is_curriculum_overview and "curriculum" in src and pstart <= 20:
            boosted_sims[idx] += 0.20
            # Pinpoint course evaluation pattern table
            if "evaluation" in q_lower and pstart == 12:
                boosted_sims[idx] += 0.35
            # Pinpoint specific semester
            for s_num in range(1, 9):
                if f"semester {s_num}" in q_lower or f"sem {s_num}" in q_lower:
                    if f"semester {s_num}" in c["text"].lower():
                        boosted_sims[idx] += 0.25

    top_indices = np.argsort(-boosted_sims)[:top_k]

    results = []
    for idx in top_indices:
        raw_score = float(sims[idx])
        if raw_score >= min_sim:
            results.append({**kb["chunks"][idx], "score": raw_score})

    return results, clean_query, corrected


# ==========================================
# CONVERSATIONAL ADVISOR PERSONA
# ==========================================

SYSTEM_PROMPT = """You are an articulate, friendly, and expert College Academic Advisor. You assist university students with accurate, clearly organized information from their official college documents.

Rules for Delivering Outstanding Answers:
1. Direct & Immediate: Answer the student's question directly in the very first sentence. Never start with disclaimers or repetitive robotic preambles.
2. Structured & Readable:
   - For Examination or Calendar dates: Group clearly by Semester (Odd / Even) or Exam type (Mid-Term vs. End-Semester). State the exact Date, Day, Month, and Semester.
   - For Courses & Credits: Present as a clean bulleted list showing: Course Code, Course Title, and Credits.
   - For Evaluation rules: Clearly state the exact percentage split (e.g., Internal 60% : External 40%) and list the continuous assessment components (CA1, CA2, MT, ES).
3. Accuracy: Synthesize facts strictly from the provided passages. Use natural inline citations like [1] or [2]. Never fabricate dates or course codes.
4. If a question asks about details not covered in the passages, explain warmly what IS available in the loaded records and advise accordingly."""


def stream_answer(question, retrieved, history):
    context_str = "\n\n".join(
        f"[{i}] ({r['source']}, {page_label(r)})\n{r['text']}"
        for i, r in enumerate(retrieved, start=1)
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for m in history[-HISTORY_TURNS:]:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append(
        {
            "role": "user",
            "content": f"OFFICIAL COLLEGE PASSAGES:\n{context_str}\n\nSTUDENT QUESTION:\n{question}",
        }
    )

    for part in ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0.2, "num_ctx": NUM_CTX},
    ):
        yield part["message"]["content"]


# ==========================================
# FRONTEND UI & STYLING
# ==========================================

st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark/Glassmorphic Look
st.markdown(
    """
    <style>
    /* Main container max width & spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(88, 28, 135, 0.35) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-bottom: 14px;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #94a3b8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 2px 4px;
    }

    /* Chat message bubble enhancements */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: rgba(30, 58, 138, 0.25);
        border-left: 3px solid #3b82f6;
    }

    /* Match Pill */
    .match-tag {
        display: inline-block;
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render Hero Banner
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎓 College Information Assistant</div>
        <div class="hero-subtitle">Instant, date-accurate academic intelligence powered by Pure Semantic RAG</div>
        <div>
            <span class="badge-pill">⚡ nomic-embed-text (768-dim)</span>
            <span class="badge-pill">🤖 Llama 3.2</span>
            <span class="badge-pill">🔍 Typo-Tolerant</span>
            <span class="badge-pill">🎯 Intent-Routed</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

check_ollama()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "kb" not in st.session_state:
    st.session_state["kb"] = build_knowledge_base(
        force=st.session_state.pop("force_rebuild", False)
    )

kb = st.session_state["kb"]

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("### 📚 Document Library")

    uploads = st.file_uploader(
        "Upload College PDFs", type="pdf", accept_multiple_files=True
    )
    if uploads and st.button("➕ Add to Library", use_container_width=True):
        os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
        for up in uploads:
            with open(os.path.join(DOCUMENT_FOLDER, up.name), "wb") as f:
                f.write(up.getbuffer())
        st.session_state.pop("kb", None)
        st.success("PDFs added! Updating semantic index...")
        st.rerun()

    if kb and kb["files"]:
        st.markdown("**Active Knowledge Base:**")
        counts = {}
        for c in kb["chunks"]:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        for name in kb["files"]:
            st.markdown(
                f"- 📄 **{name}**  \n  <small style='color: #94a3b8;'>{counts.get(name, 0)} structured passages</small>",
                unsafe_allow_html=True,
            )
        for note in kb.get("skipped", []):
            st.warning(f"⚠️ {note}")

    if st.button("🔄 Rebuild Semantic Index", use_container_width=True):
        st.session_state["force_rebuild"] = True
        st.session_state.pop("kb", None)
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ Search Controls")
    st.session_state["top_k"] = st.slider(
        "Passages to Retrieve",
        min_value=2,
        max_value=8,
        value=st.session_state.get("top_k", DEFAULT_TOP_K),
        help="Number of context passages passed to the LLM.",
    )
    st.session_state["min_sim"] = st.slider(
        "Relevance Threshold",
        min_value=0.25,
        max_value=0.65,
        value=float(st.session_state.get("min_sim", SIMILARITY_THRESHOLD)),
        step=0.05,
        help="Filters out low-relevance passages.",
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.markdown("### ✨ About This Project")
    st.caption(
        "Built for portfolio & technical demonstration.\n\n"
        "Features dense vector search with nomic embeddings, "
        "Damerau-Levenshtein typo tolerance, and "
        "intent-routed table extraction."
    )


# ==========================================
# MAIN INTERACTION LOGIC
# ==========================================

def show_sources(sources):
    if not sources:
        return
    unique_files = sorted(list(set(s["source"] for s in sources)))
    with st.expander(f"📚 Verified Sources ({len(unique_files)} document{'s' if len(unique_files) != 1 else ''} matched)"):
        for i, s in enumerate(sources, start=1):
            relevance = int(round(s.get("score", 0) * 100))
            st.markdown(
                f"**[{i}] 📄 `{s['source']}`** — *{page_label(s)}*  "
                f"<span class='match-tag'>Semantic Match: {relevance}%</span>",
                unsafe_allow_html=True,
            )
            snippet = s["text"][:380] + ("..." if len(s["text"]) > 380 else "")
            st.caption(snippet)


if kb is None or not kb.get("files"):
    st.warning("No PDF documents found in library.")
    st.info(f"Add college PDF documents into the `{DOCUMENT_FOLDER}/` directory or upload them via the sidebar.")
    st.stop()
elif not kb.get("chunks"):
    st.error("No extractable text was found in the PDF files.")
    for note in kb.get("skipped", []):
        st.write("•", note)
    st.stop()
else:
    # Render previous messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    show_sources(msg["sources"])
                if msg.get("seconds") is not None:
                    st.caption(f"⏱️ Response generated in {fmt_time(msg['seconds'])}")

    # Interactive Starter Cards
    pending_query = None
    if not st.session_state["messages"]:
        st.markdown("**💡 Popular Student Inquiries:**")
        c1, c2 = st.columns(2)
        if c1.button("📅 When are the B.Tech mid-term & semester exam dates?", use_container_width=True):
            pending_query = "What are the exact dates for mid-term and semester examinations for B.Tech?"
        if c2.button("📚 What subjects and credits are in Semester 3?", use_container_width=True):
            pending_query = "What subjects and credits are offered in Semester 3 of Mechanical Engineering?"
        if c1.button("📊 How does the 60:40 course evaluation work?", use_container_width=True):
            pending_query = "How are courses evaluated between internal Continuous Assessment (CA) and end-semester exams?"
        if c2.button("🛠️ What specializations are available in Mechanical?", use_container_width=True):
            pending_query = "What specializations are offered by the Mechanical Engineering department?"

    user_input = st.chat_input("Ask about exam dates, semester courses, evaluation rules, or holidays...") or pending_query

    if user_input:
        history = list(st.session_state["messages"])
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            start_clock = time.time()

            if GREETING_RE.match(user_input):
                greeting = (
                    f"Hello! 👋 I'm your College Academic Advisor. I have full access to your official college documents: "
                    f"**{', '.join(kb['files'])}**.\n\n"
                    "Feel free to ask me anything about semester commencement, mid-term & end-term exam dates, course syllabi, or grading policies!"
                )
                st.markdown(greeting)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": greeting, "sources": [], "seconds": None}
                )
            else:
                retrieved = []
                with st.status("Consulting college documents...", expanded=False) as status:
                    st.write("🔍 Routing query intent and analyzing semantics...")
                    t_retrieval = time.time()
                    retrieved, clean_q, was_corrected = retrieve_semantic_context(
                        user_input,
                        kb,
                        top_k=st.session_state.get("top_k", DEFAULT_TOP_K),
                        min_sim=st.session_state.get("min_sim", SIMILARITY_THRESHOLD),
                    )
                    retrieval_sec = time.time() - t_retrieval

                    if was_corrected:
                        st.info(f"💡 Auto-corrected query: *\"{clean_q}\"*")

                    if retrieved:
                        best_match = max(r["score"] for r in retrieved)
                        match_pct = int(round(best_match * 100))
                        st.write(f"✅ Found {len(retrieved)} high-relevance passage(s) (Peak Match: {match_pct}%) in {fmt_time(retrieval_sec)}.")
                        safe_update_status(
                            status,
                            label=f"Relevant records verified ({match_pct}% match) — composing response...",
                            state="running",
                            expanded=False,
                        )
                    else:
                        st.write("ℹ️ No passages met the semantic relevance threshold.")
                        safe_update_status(
                            status,
                            label="No matching college records found.",
                            state="complete",
                            expanded=False,
                        )

                if not retrieved:
                    doc_list = "\n".join(f"- `{f}`" for f in kb["files"])
                    answer = (
                        f"I couldn't locate specific information addressing that in your currently indexed documents:\n{doc_list}\n\n"
                        f"💡 **Topics you can ask about right now:**\n"
                        f"- **Examination & Commencement Dates:** Ask about B.Tech, M.Tech, or Int. M.Sc schedules.\n"
                        f"- **Curriculum & Credits:** Ask for Semester 1 to 8 course lists and credit breakdowns.\n"
                        f"- **Evaluation Policies:** Ask how internal Continuous Assessment (60%) and End-Sem (40%) exams are calculated."
                    )
                    st.markdown(answer)
                    sources_to_save = []
                else:
                    try:
                        answer = st.write_stream(stream_answer(clean_q, retrieved, history))
                    except Exception as e:
                        answer = f"An error occurred while generating the response: {e}"
                        st.error(answer)

                    show_sources(retrieved)
                    sources_to_save = retrieved

                elapsed_time = time.time() - start_clock
                st.caption(f"⏱️ Response generated in {fmt_time(elapsed_time)}")
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_to_save,
                        "seconds": elapsed_time,
                    }
                )
