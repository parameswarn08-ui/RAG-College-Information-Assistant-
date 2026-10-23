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
import inspect
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
NUM_CTX = 8192            # 4096 was too small: 5 whole curriculum pages + prompt overflowed and got truncated
HISTORY_TURNS = 4         # number of previous chat *messages* sent to the LLM (~2 exchanges)
MAX_HISTORY_CHARS = 1200  # per history message, keeps the context window for the passages
MAX_CONTEXT_CHARS = 14000 # total passage text sent to the LLM

# Bump this whenever chunking/parsing logic changes so old caches are rebuilt automatically.
CACHE_VERSION = "v5_structured_calendar"

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

MONTH_ALIASES = {
    "january": "January", "jan": "January",
    "february": "February", "feb": "February",
    "march": "March", "mar": "March",
    "april": "April", "apr": "April",
    "may": "May",
    "june": "June", "jun": "June",
    "july": "July", "jul": "July",
    "august": "August", "aug": "August",
    "september": "September", "sep": "September", "sept": "September",
    "october": "October", "oct": "October",
    "november": "November", "nov": "November",
    "december": "December", "dec": "December",
}

DAY_NAMES = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
    "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday",
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

# Ordinary words that must never be "corrected" into a vocabulary term.
COMMON_WORDS = {
    "these", "those", "there", "where", "which", "about", "after", "before", "could",
    "would", "should", "their", "dates", "offer", "offered", "offers", "exact", "terms",
    "times", "other", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "years", "final", "month", "months", "weeks", "start", "starts", "begin",
    "begins", "ending", "years", "today", "tomorrow", "please", "explain", "between",
    "number", "total", "marks", "units", "topics", "write", "being", "great", "major",
    "minor", "level", "types", "scheme", "pattern", "works", "while", "using", "under",
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


def normalize_query(query, vocabulary, known=frozenset()):
    """
    Correct likely typos against the domain vocabulary.

    - `known` = every word that occurs in the indexed documents. Those are real words and are
      never "corrected" (otherwise "external" -> "internal", "technical" -> "mechanical", ...).

    - Edits words in place, so the user's punctuation/spacing is preserved
      (the old tokenizer turned "B.Tech" into "B. mtech" / "B. btech").
    - Only words of 5+ letters are considered, and never words that follow a "."
      (acronyms such as B.Tech, M.Sc).
    - Deterministic: ties are broken alphabetically (set order used to be random).
    """
    vocab_sorted = sorted(vocabulary)
    changed = False

    def _fix(match):
        nonlocal changed
        tok = match.group(0)
        low = tok.lower()
        if low in vocabulary or low in COMMON_WORDS or low in known:
            return tok
        allowed = 1 if len(low) <= 7 else 2
        best, best_d = None, allowed + 1
        for term in vocab_sorted:
            if abs(len(term) - len(low)) > allowed:
                continue
            dist = edit_distance(low, term)
            if dist < best_d:
                best, best_d = term, dist
        if best is None:
            return tok
        changed = True
        return best.capitalize() if tok[0].isupper() else best

    fixed = re.sub(r"(?<![\w.])[A-Za-z]{5,}(?!\w)", _fix, query)
    return fixed.strip(), changed


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
    r"^\s*(hi|hello|hey|hii+|good\s*(morning|afternoon|evening)|thanks|thank\s*you|thx|ok|okay)\W*$",
    re.I,
)
THANKS_RE = re.compile(r"thank|thx", re.I)

# ==========================================
# STRUCTURED PARSING & CHUNKING
# ==========================================

def list_pdfs():
    os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
    return sorted(
        f for f in os.listdir(DOCUMENT_FOLDER) if f.lower().endswith(".pdf")
    )


# --- Academic calendar ------------------------------------------------------

# "18 Wed W CD 1 Commencement ..."  /  "16 Sat Janmashtami ( ... )"  /  "31 Sun H"
DAY_ROW_RE = re.compile(
    r"^(\d{1,2})\s+(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b\s*(?:([WH])\b)?\s*(.*)$"
)
FOOTER_RE = re.compile(r"^(Academic Calendar\b|[A-Z][a-z]{2}-\d{2}(Date Day Status)?$)")
MONTH_LABEL_RE = re.compile(r"\b([A-Z][a-z]{2}-\d{2})\b")
TIMETABLE_RE = re.compile(r"(\w+day)['’]s\s*Time\s*Table\s*/?", re.I)
MILESTONE_RE = re.compile(r"commenc|mid semester|end semester|last working|examination", re.I)

# Typos that exist in the source PDF itself. Without these, whole events were being dropped.
CALENDAR_TYPO_FIXES = [
    (re.compile(r"Coomencement", re.I), "Commencement"),
    (re.compile(r"SemesterExaminatins", re.I), "Semester Examinations"),
    (re.compile(r"Examinatins", re.I), "Examinations"),
    (re.compile(r"Semenster", re.I), "Semester"),
]


def _clean_event_text(raw):
    s = re.sub(r"\bCD\s*\d*\b", " ", raw)  # strip the "CD 12" working-day counters
    for pattern, repl in CALENDAR_TYPO_FIXES:
        s = pattern.sub(repl, s)
    return re.sub(r"\s+", " ", s).strip()


def _calendar_month(text, page_no):
    m = MONTH_LABEL_RE.search(text)
    if m and m.group(1) in MONTHS_MAP:
        return MONTHS_MAP[m.group(1)]
    for m_code, (m_name, year) in MONTHS_MAP.items():
        if m_code in text:
            return m_name, year
    return f"Page {page_no}", "2025-2026"


def _calendar_rows(text):
    """Split a calendar page into day rows. Text that wraps onto following lines
    (e.g. 'Last Working day for Second semester' / 'Int M.Sc /B.Sc') is attached to its row."""
    rows = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        m = DAY_ROW_RE.match(line)
        if m:
            day, dow, status, rest = m.groups()
            rows.append({"day": int(day), "dow": dow, "status": status or "", "parts": [rest]})
            continue
        if not rows:
            continue  # column headers before the first row
        if FOOTER_RE.match(line):
            continue
        rows[-1]["parts"].append(line)
    return rows


def parse_academic_calendar(reader, filename):
    chunks, milestones = [], []
    total_pages = len(reader.pages)

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        m_name, year = _calendar_month(text, page_no)
        short = m_name[:3]

        events, generic_holidays, swaps = [], [], []
        for row in _calendar_rows(text):
            event = _clean_event_text(" ".join(row["parts"]))
            day_full = DAY_NAMES.get(row["dow"], row["dow"])

            tt = TIMETABLE_RE.search(event)
            if tt:
                swaps.append(f"{short} {row['day']} ({row['dow']}) follows {tt.group(1).capitalize()}'s timetable")
                event = _clean_event_text(TIMETABLE_RE.sub(" ", event)).strip(" /")

            event = event.strip(" /")
            if not event or event.lower() == "holiday":
                if row["status"] == "H":
                    generic_holidays.append(f"{short} {row['day']} ({row['dow']})")
                continue

            events.append(f"• {m_name} {row['day']}, {year} ({day_full}): {event}")
            if MILESTONE_RE.search(event) and not event.lower().startswith("holiday"):
                milestones.append(f"• {m_name} {row['day']}, {year} ({day_full}): {event}")

        lines = [f"[Document: Academic Calendar AY 2025–2026 | Month: {m_name} {year} (Page {page_no})]"]
        if events:
            lines.append("Academic events and named holidays:")
            lines.extend(events)
        else:
            lines.append("No named events listed this month.")
        if generic_holidays:
            lines.append("Other holidays (weekends / unnamed): " + ", ".join(generic_holidays))
        if swaps:
            lines.append("Timetable adjustments: " + "; ".join(swaps))

        chunks.append(
            {
                "text": "\n".join(lines),
                "source": filename,
                "page_start": page_no,
                "page_end": page_no,
                "kind": "calendar_month",
            }
        )

    # One consolidated chunk so "when are the exams / when does the semester start" questions
    # don't depend on retrieving the right 4-5 of 14 monthly pages.
    if milestones:
        summary = (
            "[Document: Academic Calendar AY 2025–2026 | Summary: semester commencement, mid-semester exams, "
            "last working days and end-semester exams for all programmes]\n"
            + "\n".join(milestones)
        )
        chunks.append(
            {
                "text": summary,
                "source": filename,
                "page_start": 1,
                "page_end": total_pages,
                "kind": "calendar_summary",
            }
        )
    return chunks


# --- Curriculum & syllabus --------------------------------------------------

SECTION_KEYWORDS = ["semester", "evaluation", "specialization", "credit structure", "electives"]


def parse_curriculum_and_syllabus(reader, filename):
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        section_title = None
        first_lines = [l.strip() for l in text.split("\n") if l.strip()][:4]
        for l in first_lines:
            # length cap: stops body sentences from being used as a "title"
            if len(l) <= 90 and any(k in l.lower() for k in SECTION_KEYWORDS):
                section_title = l
                break
        section = f"{section_title} (Page {page_no})" if section_title else f"Page {page_no}"

        # Curriculum master tables (pages 1 to 20): keep page whole
        if page_no <= 20:
            chunk_text = f"[Document: B.Tech Mechanical Engineering Curriculum | Section: {section}]\n{text}"
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
            header = f"[Document: Course Syllabus | Section: {section}]\n"
            if len(words) <= 350:
                chunks.append(
                    {
                        "text": header + text,
                        "source": filename,
                        "page_start": page_no,
                        "page_end": page_no,
                    }
                )
            else:
                start = 0
                while start < len(words):
                    sub_text = " ".join(words[start : start + 300])
                    chunks.append(
                        {
                            "text": header + sub_text,
                            "source": filename,
                            "page_start": page_no,
                            "page_end": page_no,
                        }
                    )
                    if start + 300 >= len(words):
                        break  # no tiny duplicate tail chunk
                    start += 250
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

def build_known_words(chunks):
    """All 5+ letter words that appear in the documents (used to protect real words from typo 'correction')."""
    known = set()
    for c in chunks:
        known.update(w.lower() for w in re.findall(r"[A-Za-z]{5,}", c["text"]))
    return known


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(files):
    # Content hash (not mtime) so the cache stays valid after a git clone / re-download.
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    for f in files:
        p = os.path.join(DOCUMENT_FOLDER, f)
        h.update(f"{f}|{os.path.getsize(p)}|{_file_sha256(p)}".encode())
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
                if len(chunks) == embeddings.shape[0]:
                    vocab = set(meta.get("vocab", [])) | BASE_VOCABULARY
                    return {
                        "chunks": chunks,
                        "embeddings": embeddings,
                        "files": files,
                        "vocab": vocab,
                        "known": build_known_words(chunks),
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

        try:
            for start in range(0, len(chunks), batch_size):
                batch = [c["text"] for c in chunks[start : start + batch_size]]
                vectors.append(embed_texts(batch, "doc"))
                done = min(start + batch_size, len(chunks))
                elapsed = time.time() - t0
                eta = (elapsed / done) * (len(chunks) - done) if done > 0 else 0
                emb_bar.progress(done / len(chunks), text=f"Embedded {done}/{len(chunks)} ({fmt_time(eta)} left)")
        except Exception as e:
            safe_update_status(status, label="Embedding failed.", state="error")
            st.error(f"Could not create embeddings: {e}")
            st.caption(f"Check that `{EMBEDDING_MODEL}` is available: `ollama pull {EMBEDDING_MODEL}`")
            st.stop()

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
        "known": build_known_words(chunks),
        "skipped": skipped,
    }


# ==========================================
# SMART INTENT-ROUTED RETRIEVAL
# ==========================================

# Whole-word matching. The old substring checks fired on e.g. "sem" in "assessment"
# and "date" in "update".
CALENDAR_RE = re.compile(
    r"\b(dates?|calendar|when|exams?|examinations?|commence\w*|holidays?|schedules?|re-?open\w*|vacations?|"
    r"last working|working days?|begin\w*|starts?|ends?|mid|mid-?terms?|camp|breaks?|time ?tables?)\b"
)
MILESTONE_QUERY_RE = re.compile(
    r"\b(commence\w*|mid|mid-?terms?|exams?|examinations?|last working|working days?|begin\w*|starts?|ends?|end)\b"
)
CURRICULUM_RE = re.compile(
    r"\b(semesters?|sems?|subjects?|courses?|credits?|evaluation|grading|internal|external|"
    r"specializations?|specialisations?|electives?|syllabus|curriculum)\b"
)
EVALUATION_RE = re.compile(r"\b(evaluat\w*|grading|marks|internal|external|assessment|ca\d?|60\s*:\s*40)\b")
CHUNK_MONTH_RE = re.compile(r"Month: ([A-Za-z]+)")


def retrieve_semantic_context(query, kb, top_k=DEFAULT_TOP_K, min_sim=SIMILARITY_THRESHOLD):
    """
    Semantic retrieval with intent routing:
    - Normalizes typos in user query.
    - Accurately routes dates to Calendar and curriculum to Master Tables (p. 1–20).
    - Prevents textbook references from drowning out core answers.
    """
    if kb is None or kb["embeddings"] is None or len(kb["chunks"]) == 0:
        return [], query, False

    vocab = kb.get("vocab", BASE_VOCABULARY)
    clean_query, corrected = normalize_query(query, vocab, kb.get("known", frozenset()))
    q_lower = clean_query.lower()

    # Fast vector embedding
    q_vec = embed_texts([clean_query], kind="query")
    sims = (kb["embeddings"] @ q_vec.T).squeeze()

    if sims.ndim == 0:
        sims = np.array([float(sims)])

    # Intent flags
    is_calendar_query = bool(CALENDAR_RE.search(q_lower))
    is_curriculum_overview = bool(CURRICULUM_RE.search(q_lower))
    is_milestone_query = bool(MILESTONE_QUERY_RE.search(q_lower))
    is_evaluation_query = bool(EVALUATION_RE.search(q_lower))
    asked_months = {name for alias, name in MONTH_ALIASES.items() if re.search(rf"\b{alias}\b", q_lower)}
    asked_semesters = [
        s for s in range(1, 9) if re.search(rf"\b(?:semester|sem)\s*{s}\b", q_lower)
    ]

    boosts = np.zeros(len(kb["chunks"]), dtype="float32")
    for idx, c in enumerate(kb["chunks"]):
        src = c["source"].lower()
        pstart = c.get("page_start", 999)
        text_lower = None

        # Prioritize Academic Calendar for date/event queries
        if is_calendar_query and "calendar" in src:
            boosts[idx] += 0.22
            m = CHUNK_MONTH_RE.search(c["text"][:160])
            if m and m.group(1) in asked_months:
                boosts[idx] += 0.25
            if c.get("kind") == "calendar_summary" and is_milestone_query:
                boosts[idx] += 0.10

        # Prioritize Curriculum Master Pages (pages 1 to 20) over individual textbook pages
        if is_curriculum_overview and "curriculum" in src and pstart <= 20:
            boosts[idx] += 0.20
            text_lower = c["text"].lower()
            # Pinpoint the course evaluation pattern table (found by content, not a hard-coded page number)
            if is_evaluation_query and "evaluation pattern" in text_lower:
                boosts[idx] += 0.35
            # Pinpoint specific semester
            for s_num in asked_semesters:
                if f"semester {s_num}" in text_lower:
                    boosts[idx] += 0.25

    ranked = sims + boosts
    top_indices = np.argsort(-ranked)[:top_k]

    results = []
    for idx in top_indices:
        raw_score = float(sims[idx])
        # Intent-routed passages get a slightly lower floor; otherwise a boost could rank a chunk
        # first and then the raw-score filter would throw it away again.
        floor = min_sim if boosts[idx] == 0 else max(0.25, min_sim - 0.12)
        if raw_score >= floor:
            results.append({**kb["chunks"][idx], "score": raw_score})

    # Keep the passages within the LLM context budget (best-ranked first), so the sources
    # shown to the user are exactly the passages the model saw.
    fitted, used = [], 0
    for r in results:
        size = len(r["text"])
        if fitted and used + size > MAX_CONTEXT_CHARS:
            break
        fitted.append(r)
        used += size

    return fitted, clean_query, corrected


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
        messages.append({"role": m["role"], "content": m["content"][:MAX_HISTORY_CHARS]})

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


def _stretch_kwargs():
    """`use_container_width` is deprecated in newer Streamlit (replaced by width="stretch").
    Use whichever this installed version supports so the app runs on both."""
    try:
        if "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


STRETCH = _stretch_kwargs()

# Custom CSS — uses translucent neutrals so it looks right in both Streamlit light and dark themes
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
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.16) 0%, rgba(168, 85, 247, 0.16) 100%);
        border: 1px solid rgba(128, 128, 128, 0.28);
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
        background: linear-gradient(90deg, #3b82f6, #a855f7);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        opacity: 0.8;
        font-size: 1.05rem;
        margin-bottom: 14px;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.28);
        color: inherit;
        opacity: 0.9;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 2px 4px;
    }

    /* Chat message bubble enhancements */
    [data-testid="stChatMessage"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(59, 130, 246, 0.10);
        border-left: 3px solid #3b82f6;
    }

    /* Match Pill */
    .match-tag {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(59, 130, 246, 0.45);
        color: inherit;
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
        <div class="hero-subtitle">Instant, date-accurate academic intelligence powered by Semantic RAG</div>
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
    if uploads and st.button("➕ Add to Library", **STRETCH):
        os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
        for up in uploads:
            safe_name = os.path.basename(up.name)  # never let an upload write outside documents/
            with open(os.path.join(DOCUMENT_FOLDER, safe_name), "wb") as f:
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
                f"- 📄 **{name}**  \n  <small style='opacity: 0.7;'>{counts.get(name, 0)} structured passages</small>",
                unsafe_allow_html=True,
            )
        for note in kb.get("skipped", []):
            st.warning(f"⚠️ {note}")

    if st.button("🔄 Rebuild Semantic Index", **STRETCH):
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

    if st.button("🗑️ Clear Conversation", **STRETCH):
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
            if msg.get("error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    show_sources(msg["sources"])
                if msg.get("seconds") is not None:
                    st.caption(f"⏱️ Response generated in {fmt_time(msg['seconds'])}")

    # Interactive Starter Cards (in a slot, so they disappear as soon as a question is asked)
    pending_query = None
    starter_slot = st.empty()
    if not st.session_state["messages"]:
        with starter_slot.container():
            st.markdown("**💡 Popular Student Inquiries:**")
            c1, c2 = st.columns(2)
            if c1.button("📅 When are the B.Tech mid-term & semester exam dates?", **STRETCH):
                pending_query = "What are the exact dates for mid-term and semester examinations for B.Tech?"
            if c2.button("📚 What subjects and credits are in Semester 3?", **STRETCH):
                pending_query = "What subjects and credits are offered in Semester 3 of Mechanical Engineering?"
            if c1.button("📊 How does the 60:40 course evaluation work?", **STRETCH):
                pending_query = "How are courses evaluated between internal Continuous Assessment (CA) and end-semester exams?"
            if c2.button("🛠️ What specializations are available in Mechanical?", **STRETCH):
                pending_query = "What specializations are offered by the Mechanical Engineering department?"

    user_input = st.chat_input("Ask about exam dates, semester courses, evaluation rules, or holidays...") or pending_query

    if user_input:
        starter_slot.empty()

        # Failed generations are shown to the user but never fed back to the model as "history".
        history = [m for m in st.session_state["messages"] if not m.get("error")]
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            start_clock = time.time()

            if GREETING_RE.match(user_input):
                if THANKS_RE.search(user_input):
                    greeting = "You're welcome! 😊 Ask me anything else about exam dates, courses, or grading policies."
                else:
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
                retrieved, clean_q, was_corrected = [], user_input, False
                retrieval_error = None
                with st.status("Consulting college documents...", expanded=False) as status:
                    st.write("🔍 Routing query intent and analyzing semantics...")
                    t_retrieval = time.time()
                    try:
                        retrieved, clean_q, was_corrected = retrieve_semantic_context(
                            user_input,
                            kb,
                            top_k=st.session_state.get("top_k", DEFAULT_TOP_K),
                            min_sim=st.session_state.get("min_sim", SIMILARITY_THRESHOLD),
                        )
                    except Exception as e:
                        retrieval_error = e
                    retrieval_sec = time.time() - t_retrieval

                    if retrieval_error is not None:
                        st.write("⚠️ Search failed.")
                        safe_update_status(status, label="Search failed.", state="error", expanded=False)
                    else:
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

                generation_failed = False
                sources_to_save = []

                if retrieval_error is not None:
                    generation_failed = True
                    answer = f"Something went wrong while searching the documents: {retrieval_error}"
                    st.error(answer)
                elif not retrieved:
                    doc_list = "\n".join(f"- `{f}`" for f in kb["files"])
                    answer = (
                        f"I couldn't locate specific information addressing that in your currently indexed documents:\n{doc_list}\n\n"
                        f"💡 **Topics you can ask about right now:**\n"
                        f"- **Examination & Commencement Dates:** Ask about B.Tech, M.Tech, or Int. M.Sc schedules.\n"
                        f"- **Curriculum & Credits:** Ask for Semester 1 to 8 course lists and credit breakdowns.\n"
                        f"- **Evaluation Policies:** Ask how internal Continuous Assessment (60%) and End-Sem (40%) exams are calculated."
                    )
                    st.markdown(answer)
                else:
                    try:
                        streamed = st.write_stream(stream_answer(user_input, retrieved, history))
                        if isinstance(streamed, (list, tuple)):
                            streamed = "".join(str(s) for s in streamed)
                        answer = (streamed or "").strip()
                        if not answer:
                            generation_failed = True
                            answer = "I couldn't generate a response. Please try rephrasing your question."
                            st.warning(answer)
                    except Exception as e:
                        generation_failed = True
                        answer = f"An error occurred while generating the response: {e}"
                        st.error(answer)

                    if not generation_failed:
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
                        "error": generation_failed,
                    }
                )
