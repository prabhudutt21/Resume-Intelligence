
import hashlib
import io
import json
import os
import re
import sqlite3
import textwrap
from contextlib import contextmanager
from datetime import date, datetime

import pandas as pd
import pymupdf
import streamlit as st

st.set_page_config(
    page_title="Resume Intelligence",
    page_icon="📄",
    layout="wide"
)

APP_VERSION = "4.2"
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "resume_parser.db"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.42


# =========================================================
# SKILLS AND EXTRACTION SETTINGS
# =========================================================

SKILLS = [
    "Python", "Java", "C++", "C", "SQL", "HTML", "CSS",
    "JavaScript", "React", "Node.js", "Machine Learning",
    "Deep Learning", "Artificial Intelligence",
    "Data Analysis", "Data Science", "Pandas", "NumPy",
    "Matplotlib", "Seaborn", "TensorFlow", "PyTorch",
    "Scikit-learn", "Excel", "Power BI", "Tableau",
    "Git", "GitHub", "AWS", "Azure", "Docker",
    "Flask", "Django", "REST API",
    "Object-Oriented Programming", "Data Structures",
    "Algorithms", "Statistics",
    "Natural Language Processing", "Computer Vision",
    "Communication", "Leadership", "Problem Solving",
    "Teamwork", "Classroom Management", "Lesson Planning",
    "Biology", "Science", "Teaching", "Botany",
    "Assessment", "Google Classroom", "MS Office",
]

SKILL_ALIASES = {
    "C++": ["c++", "cpp"],
    "C": ["programming in c", "c language"],
    "Machine Learning": ["machine learning", "ml"],
    "Artificial Intelligence": ["artificial intelligence", " ai "],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "REST API": ["rest api", "restful api", "rest apis"],
    "Object-Oriented Programming": [
        "object-oriented programming",
        "object oriented programming",
        "oop",
    ],
    "Data Structures": ["data structures", "data structure"],
    "Algorithms": ["algorithms", "algorithm"],
    "Natural Language Processing": [
        "natural language processing", "nlp"
    ],
    "MS Office": ["ms office", "microsoft office"],
    "Problem Solving": ["problem solving", "problem-solving"],
}

EDUCATION_TERMS = [
    "B.Tech", "B.E.", "B.Sc", "B.Com", "BCA", "BBA",
    "M.Tech", "MCA", "MBA", "M.Sc", "M.Com",
    "Bachelor", "Master", "Ph.D", "Diploma",
    "B.Ed", "M.Ed", "OTET", "OSSTET",
    "Graduation", "Post Graduation",
]

EDUCATION_HEADINGS = [
    "education",
    "educational qualification",
    "educational qualifications",
    "academic qualification",
    "academic qualifications",
    "academic details",
    "education details",
    "academic background",
]

EXPERIENCE_HEADINGS = [
    "work experience",
    "professional experience",
    "employment history",
    "work history",
    "career history",
    "professional background",
]

PROJECT_HEADINGS = [
    "projects",
    "academic projects",
    "personal projects",
    "key projects",
]

STOP_HEADINGS = [
    "education",
    "education details",
    "educational qualifications",
    "academic qualifications",
    "academic details",
    "skills",
    "technical skills",
    "projects",
    "certifications",
    "achievements",
    "awards",
    "references",
    "personal details",
    "declaration",
    "languages",
    "interests",
    "hobbies",
    "profile",
    "professional summary",
    "objective",
    "contact",
    "technical expertise",
    "areas of expertise",
]

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

DATE_PATTERN = re.compile(
    rf"\b(?:{MONTH_PATTERN}\s+\d{{4}}|\d{{1,2}}[/-]\d{{4}}|\d{{4}})"
    rf"\s*(?:to|[-–—])\s*"
    rf"(?:{MONTH_PATTERN}\s+\d{{4}}|\d{{1,2}}[/-]\d{{4}}|"
    rf"\d{{4}}|Present|Current|Till Date|To Date)\b",
    re.IGNORECASE,
)


# =========================================================
# DATABASE
# =========================================================

@contextmanager
def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sha256 TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                name TEXT,
                email TEXT,
                phone TEXT,
                skills_json TEXT,
                education_json TEXT,
                experience_json TEXT,
                projects_json TEXT,
                experience_months INTEGER DEFAULT 0,
                resume_text TEXT,
                status TEXT DEFAULT 'New',
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def save_candidate(candidate):
    with db_connect() as conn:
        existing = conn.execute(
            "SELECT id, status, notes FROM candidates WHERE sha256=?",
            (candidate["sha256"],)
        ).fetchone()

        if existing:
            return (
                existing["id"],
                True,
                existing["status"],
                existing["notes"]
            )

        cur = conn.execute("""
            INSERT INTO candidates (
                sha256, filename, name, email, phone,
                skills_json, education_json, experience_json,
                projects_json, experience_months, resume_text,
                status, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'New', '', ?)
        """, (
            candidate["sha256"],
            candidate["filename"],
            candidate["name"],
            candidate["email"],
            candidate["phone"],
            json.dumps(candidate["skills"]),
            json.dumps(candidate["education"]),
            json.dumps(candidate["experience"]),
            json.dumps(candidate["projects"]),
            candidate["experience_months"],
            candidate["text"],
            datetime.now().isoformat(timespec="seconds"),
        ))

        conn.commit()

        return cur.lastrowid, False, "New", ""


def update_candidate_review(candidate_id, status, notes):
    with db_connect() as conn:
        conn.execute(
            "UPDATE candidates SET status=?, notes=? WHERE id=?",
            (status, notes, candidate_id)
        )
        conn.commit()


def delete_candidate(candidate_id):
    """Permanently delete one candidate's database record."""
    with db_connect() as conn:
        result = conn.execute(
            "DELETE FROM candidates WHERE id=?",
            (candidate_id,)
        )
        conn.commit()
        return result.rowcount > 0


def search_saved_candidates(query=""):
    with db_connect() as conn:
        if query.strip():
            needle = f"%{query.strip().lower()}%"

            rows = conn.execute("""
                SELECT id, sha256, name, email, filename, status,
                       notes, created_at, skills_json
                FROM candidates
                WHERE lower(name) LIKE ?
                   OR lower(email) LIKE ?
                   OR lower(skills_json) LIKE ?
                ORDER BY id DESC
                LIMIT 100
            """, (needle, needle, needle)).fetchall()

        else:
            rows = conn.execute("""
                SELECT id, sha256, name, email, filename, status,
                       notes, created_at, skills_json
                FROM candidates
                ORDER BY id DESC
                LIMIT 100
            """).fetchall()

    return [dict(row) for row in rows]


# =========================================================
# TEXT AND NLP HELPERS
# =========================================================

def normalize(value):
    return re.sub(
        r"[^a-z0-9+#]+",
        " ",
        str(value).lower()
    ).strip()


def has_phrase(text, phrase):
    source = normalize(text)
    target = normalize(phrase)
    return bool(target) and f" {target} " in f" {source} "


def find_matches(text, keywords):
    found = []

    for skill in keywords:
        aliases = SKILL_ALIASES.get(skill, [skill])

        if any(has_phrase(text, alias) for alias in aliases):
            found.append(skill)

    return found


def clean_line(line):
    return re.sub(
        r"^[\s•▪●*\-–—]+",
        "",
        str(line)
    ).strip()


def is_heading(line, headings):
    cleaned = normalize(
        re.sub(r"[:|]+$", "", str(line))
    )
    return cleaned in {normalize(x) for x in headings}


def extract_section(lines, headings, stop_headings):
    start = None

    for i, line in enumerate(lines):
        if is_heading(line, headings):
            start = i + 1
            break

    if start is None:
        return []

    end = len(lines)

    for i in range(start, len(lines)):
        if is_heading(lines[i], stop_headings):
            end = i
            break

    return [
        clean_line(x)
        for x in lines[start:end]
        if clean_line(x)
    ]


def extract_name(lines):
    ignored = {
        "resume", "curriculum vitae", "profile",
        "professional summary", "objective",
        "education", "skills", "experience", "contact"
    }

    for line in lines[:15]:
        candidate = line.strip(" :-|•")

        if (
            not candidate
            or "@" in candidate
            or re.search(r"\d", candidate)
        ):
            continue

        if normalize(candidate) in ignored:
            continue

        words = candidate.split()

        if (
            2 <= len(words) <= 5
            and re.fullmatch(r"[A-Za-zÀ-ÿ .'-]+", candidate)
        ):
            return candidate

    return "Not found"


def extract_email(text):
    match = re.search(
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        text
    )
    return match.group(0) if match else "Not found"


def extract_phone(text):
    candidates = re.findall(
        r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){9,12}(?!\d)",
        text
    )

    for candidate in candidates:
        digits = re.sub(r"\D", "", candidate)

        if 10 <= len(digits) <= 13:
            return candidate.strip()

    return "Not found"


# =========================================================
# EDUCATION
# =========================================================

def extract_education(lines):
    section = extract_section(
        lines,
        EDUCATION_HEADINGS,
        STOP_HEADINGS + EXPERIENCE_HEADINGS + PROJECT_HEADINGS
    )

    results = []
    seen = set()

    for line in section:
        if (
            not find_matches(line, EDUCATION_TERMS)
            or len(line.split()) > 24
        ):
            continue

        key = normalize(line)

        if key and key not in seen:
            results.append(line)
            seen.add(key)

    return results


# =========================================================
# EXPERIENCE AND PROJECTS
# =========================================================

def extract_experience(lines):
    section = extract_section(
        lines,
        EXPERIENCE_HEADINGS,
        STOP_HEADINGS + EDUCATION_HEADINGS + PROJECT_HEADINGS
    )

    if not section:
        return []

    date_positions = [
        i for i, line in enumerate(section)
        if DATE_PATTERN.search(line)
    ]

    if not date_positions:
        return [{
            "title": "Experience Details",
            "organization": "Not specified",
            "type": "Not specified",
            "dates": "Not specified",
            "description": section
        }]

    entries = []

    for idx, date_index in enumerate(date_positions):
        prev_date = date_positions[idx - 1] if idx else -1

        next_date = (
            date_positions[idx + 1]
            if idx + 1 < len(date_positions)
            else len(section)
        )

        metadata = [
            line
            for line in section[prev_date + 1:date_index]
            if line and not line.startswith(("-", "•", "▪"))
        ][-3:]

        title = metadata[0] if metadata else "Role not specified"
        organization = (
            metadata[1] if len(metadata) > 1 else "Not specified"
        )
        employment_type = (
            metadata[2] if len(metadata) > 2 else "Not specified"
        )

        details = section[date_index + 1:next_date]

        if idx + 1 < len(date_positions):
            next_metadata = [
                line for line in details
                if line and not line.startswith(("-", "•", "▪"))
            ][-3:]

            if next_metadata:
                details = details[:len(details) - len(next_metadata)]

        cleaned = []
        seen = set()

        for detail in details:
            key = normalize(detail)

            if key and key not in seen:
                cleaned.append(detail)
                seen.add(key)

        entries.append({
            "title": title,
            "organization": organization,
            "type": employment_type,
            "dates": section[date_index],
            "description": cleaned
        })

    return entries


def extract_projects(lines):
    section = extract_section(
        lines,
        PROJECT_HEADINGS,
        STOP_HEADINGS + EDUCATION_HEADINGS + EXPERIENCE_HEADINGS
    )

    cleaned = []
    seen = set()

    for line in section:
        if len(line.split()) < 2:
            continue

        key = normalize(line)

        if key and key not in seen:
            cleaned.append(line)
            seen.add(key)

    return cleaned


# =========================================================
# EXPERIENCE DURATION
# =========================================================

def parse_date_part(value):
    value = value.strip().lower()

    if value in {
        "present", "current", "till date", "to date"
    }:
        return date.today().year, date.today().month

    month_match = re.search(
        r"([a-z]+)\s+(\d{4})",
        value
    )

    if month_match:
        month_name = month_match.group(1)
        month = (
            MONTHS.get(month_name)
            or MONTHS.get(month_name[:3], 1)
        )
        return int(month_match.group(2)), month

    year_match = re.search(
        r"\b(19\d{2}|20\d{2})\b",
        value
    )

    if year_match:
        return int(year_match.group(1)), 1

    return None


def experience_duration_months(entries):
    intervals = []

    for entry in entries:
        match = DATE_PATTERN.search(entry.get("dates", ""))

        if not match:
            continue

        parts = re.split(
            r"\s*(?:to|[-–—])\s*",
            match.group(0),
            flags=re.IGNORECASE
        )

        if len(parts) != 2:
            continue

        start = parse_date_part(parts[0])
        end = parse_date_part(parts[1])

        if not start or not end:
            continue

        start_index = start[0] * 12 + start[1]
        end_index = end[0] * 12 + end[1]

        if end_index >= start_index:
            intervals.append((start_index, end_index))

    if not intervals:
        return 0

    intervals.sort()
    merged = []

    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return sum(
        end - start + 1
        for start, end in merged
    )


def duration_label(months):
    years, remaining = divmod(int(months or 0), 12)

    if years and remaining:
        return f"{years} year(s), {remaining} month(s)"

    if years:
        return f"{years} year(s)"

    if remaining:
        return f"{remaining} month(s)"

    return "Not enough date information"


# =========================================================
# PDF EXTRACTION AND OPTIONAL OCR
# =========================================================

def extract_pdf_text(raw_bytes):
    try:
        document = pymupdf.open(
            stream=raw_bytes,
            filetype="pdf"
        )
    except Exception as exc:
        raise ValueError(
            "This is not a valid or readable PDF."
        ) from exc

    try:
        pages_text = [
            page.get_text("text")
            for page in document
        ]

        text = "\n".join(pages_text).strip()

        if len(re.sub(r"\s", "", text)) >= 35:
            return text, "Text extracted from PDF"

        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise ValueError(
                "This appears to be a scanned PDF. Install pytesseract "
                "and pillow, then install Tesseract OCR for Windows."
            ) from exc

        default_tesseract = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        if os.path.exists(default_tesseract):
            pytesseract.pytesseract.tesseract_cmd = default_tesseract

        ocr_pages = []

        for page_index in range(min(len(document), 15)):
            page = document[page_index]

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False
            )

            image = Image.open(
                io.BytesIO(pix.tobytes("png"))
            )

            ocr_pages.append(
                pytesseract.image_to_string(image)
            )

        ocr_text = "\n".join(ocr_pages).strip()

        if len(re.sub(r"\s", "", ocr_text)) < 20:
            raise ValueError(
                "OCR could not read enough text from this PDF. "
                "Try a clearer scan."
            )

        return ocr_text, "Text extracted using OCR"

    except Exception as exc:
        if isinstance(exc, ValueError):
            raise

        raise ValueError(
            f"PDF text/OCR extraction failed: {exc}"
        ) from exc

    finally:
        document.close()


def process_resume(uploaded_pdf):
    raw = uploaded_pdf.getvalue()

    if not raw:
        raise ValueError("The uploaded PDF is empty.")

    text, extraction_method = extract_pdf_text(raw)

    text = (
        text.replace("\u2022", "-")
        .replace("\uf0b7", "-")
        .replace("\u00a0", " ")
        .replace("\u2011", "-")
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    experience = extract_experience(lines)

    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "name": extract_name(lines),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "filename": uploaded_pdf.name,
        "skills": find_matches(text, SKILLS),
        "education": extract_education(lines),
        "experience": experience,
        "projects": extract_projects(lines),
        "experience_months": experience_duration_months(experience),
        "text": text,
        "extraction_method": extraction_method,
    }


# =========================================================
# SEMANTIC AI MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def split_resume_into_chunks(text):
    raw_chunks = re.split(
        r"\n+|(?<=[.!?])\s+",
        text
    )

    chunks = []

    for chunk in raw_chunks:
        chunk = re.sub(
            r"\s+",
            " ",
            chunk
        ).strip(" •\t-–—")

        if 5 <= len(chunk.split()) <= 55 and len(chunk) > 25:
            chunks.append(chunk)

    return chunks[:250]


# =========================================================
# JOB REQUIREMENT FILTERING
# =========================================================

def extract_job_requirements(job_description):
    included_headings = {
        "about the role", "role overview", "job summary",
        "position summary", "job description",
        "responsibilities", "key responsibilities",
        "role and responsibilities", "what you will do",
        "what you ll do", "duties", "required skills",
        "technical skills", "preferred skills", "requirements",
        "job requirements", "qualifications",
        "required qualifications", "minimum qualifications",
        "preferred qualifications", "eligibility", "who you are",
        "what we are looking for", "what we re looking for",
        "experience requirements", "required experience",
    }

    excluded_headings = {
        "about the company", "company overview", "about us",
        "what we offer", "benefits", "perks", "compensation",
        "salary", "how to apply", "application process",
        "how to apply for this role", "equal opportunity",
        "equal employment opportunity", "contact details",
        "company details", "additional information", "why join us",
    }

    metadata_pattern = re.compile(
        r"^(?:company(?: name)?|employer|organization|organisation|"
        r"location|job location|work location|duration|internship duration|"
        r"job type|employment type|type of employment|salary|stipend|"
        r"compensation|work mode|mode of work|number of openings|openings|"
        r"application deadline|apply by|start date|end date|website|"
        r"email|phone|contact|address|reporting to)\s*[:：-]",
        re.IGNORECASE,
    )

    ignored_section_cues = re.compile(
        r"^(?:company|employer|organization|organisation|location|"
        r"duration|internship duration|job type|employment type|salary|"
        r"stipend|compensation|benefits|perks|how to apply|application process)\b",
        re.IGNORECASE,
    )

    requirement_cues = re.compile(
        r"\b(?:responsible for|responsibilities include|we are looking for|"
        r"we re looking for|candidate should|candidates should|must have|"
        r"should have|required to|ability to|experience with|experience in|"
        r"knowledge of|understanding of|familiar with|proficient in|"
        r"proficiency in|develop|build|design|implement|create|maintain|"
        r"debug|test|analy[sz]e|process|evaluate|train|deploy|integrate|"
        r"support|collaborate|assist|participate|document|communicate|"
        r"use|apply|degree|bachelor|master|diploma|graduate|graduation|"
        r"pursuing|student|coursework|programming|machine learning|"
        r"artificial intelligence|data analysis|data science|python|java|"
        r"sql|react|javascript|tensorflow|pytorch|scikit.learn|pandas|numpy)\b",
        re.IGNORECASE,
    )

    requirements = []
    seen = set()

    def add_requirement(raw_line, force=False):
        line = re.sub(
            r"^\s*#{1,6}\s*",
            "",
            str(raw_line)
        ).strip()

        line = re.sub(
            r"^\s*(?:[-*•]|\d+[.)])\s*",
            "",
            line
        )

        line = re.sub(r"[*_`]+", "", line).strip()
        line = clean_line(line)

        if not line:
            return

        if metadata_pattern.search(line):
            return

        if ignored_section_cues.search(line):
            return

        normalized = normalize(line)
        words = line.split()

        if len(words) < 4 or len(words) > 45 or len(line) < 22:
            return

        if re.fullmatch(r"[\d\s./–—-]+", line):
            return

        if not force and not requirement_cues.search(line):
            return

        if normalized and normalized not in seen:
            requirements.append(line)
            seen.add(normalized)

    active_section = "fallback"

    for raw_line in job_description.splitlines():
        line = re.sub(
            r"^\s*#{1,6}\s*",
            "",
            raw_line
        ).strip()

        line = re.sub(r"[*_`]+", "", line).strip()
        heading = normalize(line.rstrip(":"))

        if heading in included_headings:
            active_section = "include"
            continue

        if heading in excluded_headings:
            active_section = "exclude"
            continue

        if metadata_pattern.search(line):
            continue

        if not line:
            continue

        if active_section == "exclude":
            continue

        if raw_line.lstrip().startswith("#") and heading:
            active_section = "fallback"
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            line
        )

        for sentence in sentences:
            if active_section == "include":
                add_requirement(sentence, force=True)
            else:
                add_requirement(sentence, force=False)

    return requirements[:40]


# =========================================================
# SEMANTIC MATCHING
# =========================================================

def semantic_match(model, candidate_text, job_description):
    requirements = extract_job_requirements(job_description)
    chunks = split_resume_into_chunks(candidate_text)

    if not requirements or not chunks:
        return {
            "score": 0.0,
            "matched_requirements": [],
            "evidence": [],
            "requirement_count": len(requirements),
            "requirements": requirements,
        }

    import numpy as np

    req_embeddings = model.encode(
        requirements,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    chunk_embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarity_matrix = np.matmul(
        req_embeddings,
        chunk_embeddings.T
    )

    evidence = []
    matched = []

    for i, requirement in enumerate(requirements):
        best_idx = int(np.argmax(similarity_matrix[i]))
        score = float(similarity_matrix[i][best_idx])

        evidence.append({
            "requirement": requirement,
            "similarity": score,
            "evidence": chunks[best_idx],
            "matched": score >= SEMANTIC_THRESHOLD,
        })

        if score >= SEMANTIC_THRESHOLD:
            matched.append(requirement)

    coverage = (
        len(matched) / len(requirements) * 100
        if requirements else 0.0
    )

    return {
        "score": coverage,
        "matched_requirements": matched,
        "evidence": sorted(
            evidence,
            key=lambda x: x["similarity"],
            reverse=True,
        ),
        "requirement_count": len(requirements),
        "requirements": requirements,
    }


# =========================================================
# CANDIDATE ANALYSIS
# =========================================================

def calculate_score(matched, required):
    if not required:
        return None

    return len(matched) / len(required) * 100


def analyze_candidate(candidate, job_description, model=None):
    required_skills = find_matches(job_description, SKILLS)

    required_education = find_matches(
        job_description,
        EDUCATION_TERMS
    )

    matched = [
        skill for skill in required_skills
        if skill in candidate["skills"]
    ]

    missing = [
        skill for skill in required_skills
        if skill not in candidate["skills"]
    ]

    candidate_education_text = " ".join(
        candidate["education"]
    )

    matched_education = [
        term for term in required_education
        if has_phrase(candidate_education_text, term)
    ]

    skill_score = calculate_score(
        matched,
        required_skills
    )

    education_score = calculate_score(
        matched_education,
        required_education
    )

    semantic = (
        semantic_match(
            model,
            candidate["text"],
            job_description
        )
        if model else None
    )

    if semantic is not None:
        components = []

        if skill_score is not None:
            components.append((skill_score, 55))

        components.append((semantic["score"], 45))

        total_weight = sum(
            weight for _, weight in components
        )

        final_score = (
            sum(
                score * weight
                for score, weight in components
            ) / total_weight
            if total_weight else 0
        )

        score_method = (
            "Overall score combines 55% recognized skill coverage "
            "and 45% semantic requirement coverage. Education and "
            "experience are shown separately."
        )

    else:
        final_score = (
            skill_score
            if skill_score is not None else 0
        )

        score_method = (
            "Semantic model was unavailable; this result uses "
            "recognized skill keyword coverage only."
        )

    result = candidate.copy()

    result.update({
        "score": final_score,
        "skill_score": skill_score,
        "semantic_score": semantic["score"] if semantic else None,
        "semantic_details": semantic,
        "education_score": education_score,
        "matched": matched,
        "missing": missing,
        "matched_education": matched_education,
        "required_skills": required_skills,
        "required_education": required_education,
        "score_method": score_method,
    })

    return result


# =========================================================
# PDF REPORT
# =========================================================

def safe_filename(name):
    cleaned = re.sub(
        r"[^\w\s-]",
        "",
        str(name),
        flags=re.UNICODE
    )

    cleaned = re.sub(
        r"[\s-]+",
        "_",
        cleaned
    ).strip("_")

    return (cleaned or "Candidate") + ".pdf"


def generate_report(candidate, job_description):
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)

    margin, y, page_width = 48, 52, 595

    def new_page():
        nonlocal page, y
        page = pdf.new_page(width=595, height=842)
        y = 52

    def ensure_space(height=18):
        nonlocal y

        if y + height > 790:
            new_page()

    def add_text(
        value="",
        bold=False,
        size=9.5,
        color=(0.15, 0.18, 0.23),
        indent=0,
    ):
        nonlocal y

        max_chars = max(
            35,
            int(
                (page_width - 2 * margin - indent)
                / (size * 0.52)
            )
        )

        wrapped = textwrap.wrap(
            str(value),
            width=max_chars,
            break_long_words=True,
        ) or [""]

        for line in wrapped:
            ensure_space(size + 9)

            page.insert_text(
                (margin + indent, y),
                line,
                fontsize=size,
                fontname="hebo" if bold else "helv",
                color=color,
            )

            y += size + 7

    def heading(value):
        nonlocal y

        ensure_space(34)
        y += 9

        page.draw_line(
            (margin, y - 5),
            (page_width - margin, y - 5),
            color=(0.78, 0.82, 0.88),
            width=0.7,
        )

        add_text(
            value.upper(),
            True,
            11,
            (0.10, 0.25, 0.43),
        )

        y += 3

    add_text("AI RESUME PARSER", True, 20, (0.10, 0.25, 0.43))
    add_text("CANDIDATE ANALYSIS REPORT", True, 11)

    add_text(
        f"Generated: {datetime.now():%d-%m-%Y %H:%M}",
        size=8,
    )

    heading("1. Candidate Information")
    add_text(f"Name: {candidate['name']}")
    add_text(f"Email: {candidate['email']}")
    add_text(f"Phone: {candidate['phone']}")
    add_text(f"Source file: {candidate['filename']}", size=8)

    add_text(
        f"PDF extraction: {candidate.get('extraction_method', 'Text extracted from PDF')}",
        size=8,
    )

    heading("2. Detected Skills")

    add_text(
        ", ".join(candidate["skills"])
        if candidate["skills"]
        else "No recognized skills detected."
    )

    heading("3. Education Details")

    if candidate["education"]:
        for item in candidate["education"]:
            add_text("• " + item, indent=5)
    else:
        add_text(
            "No education details found in a dedicated education section."
        )

    heading("4. Work Experience")

    if candidate["experience"]:
        add_text(
            "Estimated total experience: "
            + duration_label(candidate["experience_months"])
        )

        for exp in candidate["experience"]:
            add_text(exp["title"], True)
            add_text(f"Organization: {exp['organization']}")
            add_text(f"Type: {exp['type']} | Dates: {exp['dates']}")

            for detail in exp["description"]:
                add_text("• " + detail, indent=6)

    else:
        add_text(
            "No separate work experience section found. "
            "Practicum and projects are not automatically classified as employment."
        )

    heading("5. Projects")

    if candidate.get("projects"):
        for item in candidate["projects"]:
            add_text("• " + item, indent=5)
    else:
        add_text("No dedicated projects section found.")

    heading("6. Job Description")

    for line in job_description.splitlines():
        line = line.strip()

        if line:
            line = re.sub(r"^\s*#{1,6}\s*", "", line)
            line = re.sub(r"^\s*[-*]\s*", "• ", line)
            add_text(line, size=8.5)

    if "score" in candidate:
        heading("7. Screening Results")

        add_text(
            f"Overall matching score: {candidate['score']:.1f}/100",
            True,
            12,
        )

        add_text(candidate["score_method"], size=8.5)

        if candidate.get("skill_score") is not None:
            add_text(
                f"Recognized skills coverage: "
                f"{candidate['skill_score']:.1f}%"
            )

        if candidate.get("semantic_score") is not None:
            add_text(
                f"Semantic requirement coverage: "
                f"{candidate['semantic_score']:.1f}%"
            )

        if candidate.get("education_score") is not None:
            add_text(
                f"Education keyword coverage: "
                f"{candidate['education_score']:.1f}%"
            )

        heading("8. Matching Skills")

        add_text(
            ", ".join(candidate["matched"])
            if candidate["matched"]
            else "No matching skills found."
        )

        heading("9. Missing Recognized Skills")

        add_text(
            ", ".join(candidate["missing"])
            if candidate["missing"]
            else "No missing recognized skills."
        )

        heading("10. Semantic Evidence")

        details = candidate.get("semantic_details")

        if details and details.get("evidence"):
            for item in details["evidence"][:8]:
                prefix = (
                    "MATCH"
                    if item["matched"]
                    else "POSSIBLE GAP"
                )

                add_text(
                    f"{prefix} ({item['similarity']:.2f}): "
                    f"{item['requirement']}",
                    True,
                    8.5,
                )

                add_text(
                    f"Resume evidence: {item['evidence']}",
                    size=8,
                )

        else:
            add_text(
                "Semantic evidence was not available for this analysis."
            )

    heading("Important Disclaimer")

    add_text(
        "This tool supports HR review; it does not make hiring decisions. "
        "Semantic similarity and keyword coverage can be imperfect. "
        "Recruiters must verify candidate claims, qualifications, "
        "experience and skill proficiency manually."
    )

    data = pdf.tobytes()
    pdf.close()

    return data


# =========================================================
# EXCEL EXPORT
# =========================================================

def make_excel_export(results):
    rows = []

    for candidate in results:
        rows.append({
            "Candidate": candidate["name"],
            "Email": candidate["email"],
            "Phone": candidate["phone"],
            "Overall Match (%)": round(candidate["score"], 1),
            "Skill Coverage (%)": (
                round(candidate["skill_score"], 1)
                if candidate.get("skill_score") is not None
                else "N/A"
            ),
            "Semantic Coverage (%)": (
                round(candidate["semantic_score"], 1)
                if candidate.get("semantic_score") is not None
                else "Unavailable"
            ),
            "Experience": duration_label(
                candidate["experience_months"]
            ),
            "Matched Skills": ", ".join(candidate["matched"]),
            "Missing Skills": ", ".join(candidate["missing"]),
            "Education": "; ".join(candidate["education"]),
            "Work Experience Entries": len(candidate["experience"]),
            "Projects": "; ".join(candidate["projects"]),
            "Review Status": candidate.get("status", "New"),
            "Resume File": candidate["filename"],
        })

    buffer = io.BytesIO()

    pd.DataFrame(rows).to_excel(
        buffer,
        index=False,
        engine="openpyxl",
    )

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# INITIALIZE DATABASE AND SESSION STATE
# =========================================================

init_db()

# This is the initialization that prevents the KeyError.
# It must be present before the resume-upload processing section.
st.session_state.setdefault("deleted_hashes", set())


# =========================================================
# DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
 --bg:#0b1120;
 --panel:#121c2e;
 --line:#26354c;
 --text:#edf3ff;
 --muted:#91a2bb;
 --blue:#4f8cff;
 --teal:#2dd4bf;
}

.stApp {
 background:radial-gradient(circle at 85% 0%,#142642 0%,#0b1120 38%);
 color:var(--text);
 font-family:'DM Sans',sans-serif;
}

[data-testid="stHeader"] {background:transparent;}

[data-testid="stSidebar"] {
 background:#0e1829;
 border-right:1px solid #22314a;
}

[data-testid="stSidebar"] * {color:#dce7f8;}

h1,h2,h3 {
 font-family:'Manrope',sans-serif !important;
 letter-spacing:-.35px;
}

h1 {font-size:2rem !important;font-weight:800 !important;}
h2 {font-size:1.35rem !important;font-weight:700 !important;}
h3 {font-size:1.05rem !important;font-weight:700 !important;}

p,label,[data-testid="stCaptionContainer"] {color:#aebdd2;}

[data-testid="stMetric"] {
 background:linear-gradient(145deg,#17263d,#111c2e);
 border:1px solid #263a56;
 border-radius:16px;
 padding:16px 18px;
 box-shadow:0 8px 24px #00000020;
}

[data-testid="stMetricLabel"] {color:#9eafc7 !important;}

[data-testid="stMetricValue"] {
 color:#f4f8ff !important;
 font-family:'Manrope',sans-serif;
 font-weight:800;
}

[data-testid="stFileUploader"] section {
 background:#121e31;
 border:1px dashed #45658f;
 border-radius:16px;
}

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div {
 background:#121e31 !important;
 color:#edf3ff !important;
 border:1px solid #2b3d58 !important;
 border-radius:12px !important;
}

.stButton > button,.stDownloadButton > button {
 background:linear-gradient(90deg,#3979ed,#4f8cff) !important;
 color:white !important;
 border:0 !important;
 border-radius:10px !important;
 font-weight:700 !important;
 min-height:42px;
 box-shadow:0 6px 18px #3979ed35;
}

.stButton > button:hover,.stDownloadButton > button:hover {
 background:linear-gradient(90deg,#2e68d2,#3979ed) !important;
}

[data-testid="stExpander"] {
 background:#111d30;
 border:1px solid #263951;
 border-radius:13px;
}

hr {border-color:#263951 !important;}

.ring-wrap {
 display:flex;
 justify-content:center;
 align-items:center;
 padding:8px 0 12px;
}

.ring {
 width:var(--ring-size);
 height:var(--ring-size);
 border-radius:50%;
 display:grid;
 place-items:center;
 background:conic-gradient(var(--ring-color) calc(var(--value)*1%),#26364d 0);
 position:relative;
}

.ring:before {
 content:'';
 position:absolute;
 width:calc(var(--ring-size) - 18px);
 height:calc(var(--ring-size) - 18px);
 background:#142137;
 border-radius:50%;
}

.ring-center {
 z-index:1;
 display:flex;
 flex-direction:column;
 align-items:center;
 line-height:1.2;
}

.ring-center strong {
 color:#f3f7ff;
 font:800 1.15rem 'Manrope',sans-serif;
}

.ring-center span {
 color:#9fb0c8;
 font-size:.68rem;
 margin-top:4px;
}

.donut-row {
 display:flex;
 align-items:center;
 gap:24px;
 padding:10px 0;
 flex-wrap:wrap;
}

.donut {
 width:150px;
 height:150px;
 border-radius:50%;
 display:grid;
 place-items:center;
 background:conic-gradient(#4f8cff var(--matched),#34445d 0);
}

.donut-hole {
 width:102px;
 height:102px;
 border-radius:50%;
 background:#142137;
 display:flex;
 flex-direction:column;
 align-items:center;
 justify-content:center;
}

.donut-hole b {
 color:#f4f8ff;
 font:800 1.3rem 'Manrope',sans-serif;
}

.donut-hole span {color:#9fb0c8;font-size:.7rem;}

.legend {
 display:flex;
 flex-direction:column;
 gap:12px;
 color:#b7c5d9;
 font-size:.85rem;
}

.legend div {display:flex;align-items:center;gap:9px;}
.legend b {color:#f4f8ff;margin-left:5px;}
.dot {width:9px;height:9px;border-radius:50%;display:inline-block;}
.dot.blue {background:#4f8cff;}
.dot.muted {background:#52627a;}
.section-note {color:#91a2bb;font-size:.83rem;}

.tag {
 display:inline-block;
 padding:5px 9px;
 margin:3px 4px 3px 0;
 border-radius:8px;
 background:#1c3455;
 color:#cfe0ff;
 font-size:.78rem;
 border:1px solid #2c4b74;
}

.small-card {
 background:#121e31;
 border:1px solid #263951;
 border-radius:14px;
 padding:15px 17px;
}

.small-muted {color:#91a2bb;font-size:.78rem;}

.score-pill {
 color:#cfe0ff;
 background:#1c3455;
 padding:5px 10px;
 border-radius:20px;
 font-weight:700;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## ◈ HR Panel")
    st.caption("RESUME ANALYSIS SYSTEM")
    st.divider()
    st.markdown("**QUICK GUIDE**")
    st.markdown("1. Upload resumes")
    st.markdown("2. Paste a job description")
    st.markdown("3. Analyse candidates")
    st.markdown("4. Review and shortlist")
    st.markdown("5. Manage saved candidates")
    st.divider()
    st.caption("PDF extraction, semantic matching, OCR and database")
    st.caption(f"Version {APP_VERSION}")


# =========================================================
# HEADER AND UPLOAD
# =========================================================

st.markdown("# Resume Intelligence")

st.markdown(
    "<div class='section-note'>Upload resumes, understand candidate evidence, compare job requirements and generate HR reports.</div>",
    unsafe_allow_html=True,
)

st.write("")

uploaded_files = st.file_uploader(
    "① Upload candidate resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload multiple text-based or scanned PDF resumes. OCR requires Tesseract."
)


# =========================================================
# PROCESS RESUMES
# =========================================================

candidates = []

if uploaded_files:
    seen_hashes = set()

    for uploaded_pdf in uploaded_files:
        try:
            candidate = process_resume(uploaded_pdf)
            candidate_hash = candidate["sha256"]

            if candidate_hash in seen_hashes:
                st.info(
                    f"Skipped repeated upload in this batch: {uploaded_pdf.name}"
                )
                continue

            seen_hashes.add(candidate_hash)

            # Do not automatically recreate a deleted record
            # while its original PDF is still in this upload widget.
            if candidate_hash in st.session_state["deleted_hashes"]:
                st.info(
                    f"{uploaded_pdf.name} was deleted from the database. "
                    "It is being skipped in this session."
                )
                continue

            candidate_id, duplicate, status, notes = save_candidate(candidate)

            candidate["candidate_id"] = candidate_id
            candidate["duplicate"] = duplicate
            candidate["status"] = status
            candidate["notes"] = notes

            candidates.append(candidate)

            if duplicate:
                st.caption(
                    f"{uploaded_pdf.name}: existing local candidate record reused."
                )

        except Exception as error:
            st.error(
                f"Could not process {uploaded_pdf.name}: {error}"
            )


# =========================================================
# CANDIDATE OVERVIEW AND PROFILES
# =========================================================

if not candidates:
    st.info(
        "Upload one or more PDF resumes to begin. "
        "Candidate records are stored locally on this computer."
    )

else:
    st.markdown("### Overview")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Resumes uploaded", len(uploaded_files))
    m2.metric("Successfully parsed", len(candidates))

    m3.metric(
        "Skills identified",
        sum(len(c["skills"]) for c in candidates)
    )

    m4.metric(
        "Experience entries",
        sum(len(c["experience"]) for c in candidates)
    )

    st.markdown("### Candidate profiles")

    for index, candidate in enumerate(candidates):
        with st.expander(
            f"{candidate['name']} · {candidate['filename']}",
            expanded=False
        ):
            left, right = st.columns([1.4, 1])

            with left:
                st.markdown("**Contact information**")
                st.write(f"Email: {candidate['email']}")
                st.write(f"Phone: {candidate['phone']}")

                st.markdown("**Education**")

                if candidate["education"]:
                    for item in candidate["education"]:
                        st.write(f"• {item}")
                else:
                    st.caption(
                        "No qualification found in a dedicated education section."
                    )

            with right:
                st.markdown("**Profile summary**")
                st.metric("Skills found", len(candidate["skills"]))
                st.metric(
                    "Experience entries",
                    len(candidate["experience"])
                )
                st.metric(
                    "Estimated experience",
                    duration_label(candidate["experience_months"])
                )

            st.markdown("**Detected skills**")

            if candidate["skills"]:
                st.markdown(
                    " ".join(
                        f"<span class='tag'>{skill}</span>"
                        for skill in candidate["skills"]
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.caption("No recognized skills found.")

            st.markdown("**Work experience**")

            if candidate["experience"]:
                for exp in candidate["experience"]:
                    st.markdown(
                        f"**{exp['title']}** · {exp['organization']}"
                    )

                    st.caption(
                        f"{exp['type']} | {exp['dates']}"
                    )

                    for detail in exp["description"]:
                        st.write(f"• {detail}")

            else:
                st.info(
                    "No separate work experience section found. "
                    "Practicum and projects are not automatically classified as employment."
                )

            st.markdown("**Projects**")

            if candidate["projects"]:
                for project in candidate["projects"]:
                    st.write(f"• {project}")
            else:
                st.caption("No dedicated projects section found.")

            st.caption(candidate["extraction_method"])

            with st.expander("View extracted resume text"):
                st.text_area(
                    "Resume text",
                    candidate["text"],
                    height=220,
                    key=f"resume_text_{index}"
                )


    # =====================================================
    # JOB DESCRIPTION
    # =====================================================

    st.markdown("### ② Job description")

    st.caption(
        "Paste the actual vacancy description, including required skills and qualifications."
    )

    job_description = st.text_area(
        "Job description",
        placeholder=(
            "Example: We are looking for a Python developer with SQL, "
            "Pandas, NumPy and communication skills..."
        ),
        height=155,
        key="job_description",
        label_visibility="collapsed"
    )

    if st.button(
        "Analyse candidates",
        type="primary",
        use_container_width=True
    ):

        if not job_description.strip():
            st.warning("Please paste a job description first.")

        else:
            model = None
            model_error = None

            try:
                with st.spinner(
                    "Loading semantic AI model. The first run may download it..."
                ):
                    model = load_embedding_model()

            except Exception as exc:
                model_error = str(exc)

            if model is None:
                st.warning(
                    "Semantic AI is unavailable, so this analysis will "
                    "use keyword matching only. Technical detail: "
                    + model_error[:250]
                )

            with st.spinner(
                "Analysing resumes against the job description..."
            ):
                results = [
                    analyze_candidate(
                        candidate,
                        job_description,
                        model=model
                    )
                    for candidate in candidates
                ]

            results.sort(
                key=lambda item: item["score"],
                reverse=True
            )

            st.session_state["results"] = results
            st.session_state["job"] = job_description

            st.session_state["analysis_mode"] = (
                "Semantic AI + keyword"
                if model
                else "Keyword fallback"
            )

            st.rerun()


    # =====================================================
    # ANALYSIS DASHBOARD
    # =====================================================

    results = st.session_state.get("results")
    saved_job = st.session_state.get("job")

    if results and saved_job == job_description:

        st.divider()
        st.markdown("### ③ Candidate analysis")

        st.caption(
            "Ranking is a screening aid, not a hiring decision. "
            "Always review candidate evidence."
        )

        st.caption(
            f"Analysis method: "
            f"{st.session_state.get('analysis_mode', 'Not specified')}"
        )

        if st.session_state.get("analysis_mode") == "Keyword fallback":
            st.warning(
                "Semantic matching was unavailable. "
                "This result uses keyword coverage only."
            )

        chart_col, list_col = st.columns([1.25, 1])

        with chart_col:
            st.markdown("**Candidate match comparison**")

            chart_data = pd.DataFrame({
                "Candidate": [c["name"] for c in results],
                "Overall match": [
                    round(c["score"], 1)
                    for c in results
                ],
            }).set_index("Candidate")

            st.bar_chart(
                chart_data,
                horizontal=True,
                height=max(210, 65 * len(results)),
                color="#4F8CFF"
            )

        with list_col:
            st.markdown("**Candidate ranking**")

            for rank, candidate in enumerate(results, 1):
                st.markdown(
                    f"""
                    <div class='small-card' style='margin-bottom:9px;'>
                      <div style='display:flex;justify-content:space-between;gap:8px;'>
                        <b>{rank}. {candidate['name']}</b>
                        <span class='score-pill'>{candidate['score']:.1f}%</span>
                      </div>
                      <div class='small-muted'>
                        {len(candidate['matched'])} recognized skills matched ·
                        {len(candidate['missing'])} missing
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.download_button(
            "Export candidate comparison to Excel",
            data=make_excel_export(results),
            file_name="Resume_Analysis_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()
        st.markdown("**Select a candidate to view details**")

        candidate_options = [
            f"{c['name']} ({c['filename']})"
            for c in results
        ]

        selected_option = st.selectbox(
            "Candidate",
            candidate_options,
            label_visibility="collapsed"
        )

        selected = results[
            candidate_options.index(selected_option)
        ]

        score_col, skills_col, semantic_col = st.columns(3)

        with score_col:
            st.markdown(
                "<div class='small-card'>"
                "<div class='small-muted' style='text-align:center;'>"
                "OVERALL MATCH</div>",
                unsafe_allow_html=True
            )

            val = selected["score"]

            st.markdown(
                f"""
                <div class='ring-wrap'>
                  <div class='ring'
                    style='--value:{val};--ring-color:#4F8CFF;--ring-size:142px'>
                    <div class='ring-center'>
                      <strong>{val:.1f}%</strong>
                      <span>overall</span>
                    </div>
                  </div>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with skills_col:
            st.markdown(
                "<div class='small-card'>"
                "<div class='small-muted' style='text-align:center;'>"
                "SKILL COVERAGE</div>",
                unsafe_allow_html=True
            )

            if selected["skill_score"] is not None:
                val = selected["skill_score"]

                st.markdown(
                    f"""
                    <div class='ring-wrap'>
                      <div class='ring'
                        style='--value:{val};--ring-color:#2DD4BF;--ring-size:122px'>
                        <div class='ring-center'>
                          <strong>{val:.1f}%</strong>
                          <span>skills</span>
                        </div>
                      </div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("No recognized skills in JD")

        with semantic_col:
            st.markdown(
                "<div class='small-card'>"
                "<div class='small-muted' style='text-align:center;'>"
                "SEMANTIC COVERAGE</div>",
                unsafe_allow_html=True
            )

            if selected["semantic_score"] is not None:
                val = selected["semantic_score"]

                st.markdown(
                    f"""
                    <div class='ring-wrap'>
                      <div class='ring'
                        style='--value:{val};--ring-color:#8B9CFF;--ring-size:122px'>
                        <div class='ring-center'>
                          <strong>{val:.1f}%</strong>
                          <span>semantic</span>
                        </div>
                      </div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Semantic model unavailable")

        st.caption(selected["score_method"])

        # SKILLS BREAKDOWN
        st.markdown("### Skills breakdown")

        match_col, detail_col = st.columns([1, 1.4])

        with match_col:
            total = (
                len(selected["matched"])
                + len(selected["missing"])
            )

            percentage = (
                len(selected["matched"]) / total * 100
                if total else 0
            )

            st.markdown(
                f"""
                <div class='donut-row'>
                  <div class='donut' style='--matched:{percentage}%;'>
                    <div class='donut-hole'>
                      <b>{percentage:.0f}%</b>
                      <span>Matched</span>
                    </div>
                  </div>
                  <div class='legend'>
                    <div><i class='dot blue'></i>Matched
                    <b>{len(selected['matched'])}</b></div>
                    <div><i class='dot muted'></i>Missing
                    <b>{len(selected['missing'])}</b></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with detail_col:
            st.markdown("**Matched skills**")

            if selected["matched"]:
                st.markdown(
                    " ".join(
                        f"<span class='tag'>{skill}</span>"
                        for skill in selected["matched"]
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.caption("No recognized matching skills found.")

            st.markdown("**Skills not found in resume text**")

            if selected["missing"]:
                st.markdown(
                    " ".join(
                        f"<span class='tag' "
                        f"style='background:#273247;border-color:#3b4b63;"
                        f"color:#c2ccda;'>{skill}</span>"
                        for skill in selected["missing"]
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.caption("No missing recognized skills.")

        # SEMANTIC EVIDENCE
        st.markdown("### Semantic requirement evidence")

        semantic_details = selected.get("semantic_details")

        if semantic_details and semantic_details.get("evidence"):
            st.caption(
                f"{semantic_details['requirement_count']} job requirements "
                "were included in semantic analysis. Company details, "
                "location, duration and benefits are excluded."
            )

            for item in semantic_details["evidence"][:10]:
                label = (
                    "Potentially supported"
                    if item["matched"]
                    else "Needs manual review"
                )

                st.markdown(
                    f"**{label} · similarity {item['similarity']:.2f}**"
                )

                st.write(
                    f"Job requirement: {item['requirement']}"
                )

                st.caption(
                    f"Closest resume evidence: {item['evidence']}"
                )

        else:
            st.info(
                "No semantic evidence was created for this analysis."
            )

        # EDUCATION AND EXPERIENCE
        details_left, details_right = st.columns(2)

        with details_left:
            st.markdown("### Education")

            if selected["education"]:
                for item in selected["education"]:
                    st.write(f"• {item}")
            else:
                st.caption(
                    "No education details found in a dedicated section."
                )

            st.markdown("**Education keyword matches**")

            st.write(
                ", ".join(selected["matched_education"])
                if selected["matched_education"]
                else "No matching education keywords found."
            )

        with details_right:
            st.markdown("### Experience evidence")

            st.write(
                f"Estimated total duration: "
                f"**{duration_label(selected['experience_months'])}**"
            )

            if selected["experience"]:
                for exp in selected["experience"]:
                    st.markdown(f"**{exp['title']}**")

                    st.caption(
                        f"{exp['organization']} · "
                        f"{exp['type']} · {exp['dates']}"
                    )

                    for detail in exp["description"]:
                        st.write(f"• {detail}")
            else:
                st.caption("No separate work experience section found.")

            st.markdown("### Projects")

            if selected["projects"]:
                for project in selected["projects"]:
                    st.write(f"• {project}")
            else:
                st.caption("No dedicated projects section found.")

        # HR REVIEW
        st.divider()
        st.markdown("### HR review")

        status_options = [
            "New",
            "Shortlisted",
            "Under Review",
            "Rejected",
        ]

        current_status_index = (
            status_options.index(selected.get("status", "New"))
            if selected.get("status", "New") in status_options
            else 0
        )

        review_status = st.selectbox(
            "Candidate status",
            status_options,
            index=current_status_index,
            key=f"status_{selected['candidate_id']}"
        )

        review_notes = st.text_area(
            "Recruiter notes",
            value=selected.get("notes", ""),
            key=f"notes_{selected['candidate_id']}",
            placeholder="Add interview notes or follow-up details..."
        )

        if st.button(
            "Save HR review",
            key=f"save_review_{selected['candidate_id']}"
        ):
            update_candidate_review(
                selected["candidate_id"],
                review_status,
                review_notes
            )

            selected["status"] = review_status
            selected["notes"] = review_notes

            st.success(
                "HR review saved in the local database."
            )

        # PDF REPORT
        st.markdown("### Candidate report")

        pdf_data = generate_report(
            selected,
            saved_job
        )

        st.download_button(
            f"Download {safe_filename(selected['name'])}",
            data=pdf_data,
            file_name=safe_filename(selected["name"]),
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )


# =========================================================
# SAVED CANDIDATE DATABASE
# =========================================================

st.divider()
st.markdown("### Saved candidate database")

st.caption(
    "Candidate records and HR review notes are stored locally "
    "in resume_parser.db on this computer."
)

search_query = st.text_input(
    "Search saved candidates by name, email or skill",
    placeholder="Search candidates..."
)

saved_candidates = search_saved_candidates(search_query)

if saved_candidates:

    display_rows = []

    for row in saved_candidates:
        display_rows.append({
            "ID": row["id"],
            "Candidate": row["name"],
            "Email": row["email"],
            "Status": row["status"],
            "Resume": row["filename"],
            "Saved": row["created_at"],
        })

    st.dataframe(
        pd.DataFrame(display_rows),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # PERMANENT DELETE OPTION
    # -----------------------------------------------------

    st.markdown("#### Manage saved candidates")

    st.caption(
        "Select a saved candidate below to permanently remove "
        "their stored record and associated extracted information."
    )

    delete_options = [
        (
            f"{row['name']} — {row['filename']} "
            f"(Database ID: {row['id']})"
        )
        for row in saved_candidates
    ]

    selected_delete_label = st.selectbox(
        "Select candidate to delete",
        delete_options,
        key="delete_candidate_selection"
    )

    selected_delete_index = delete_options.index(
        selected_delete_label
    )

    selected_delete_record = saved_candidates[
        selected_delete_index
    ]

    st.warning(
        f"You selected **{selected_delete_record['name']}** "
        f"from file **{selected_delete_record['filename']}**."
    )

    confirm_delete = st.checkbox(
        "I understand this permanently deletes this candidate's "
        "database record, extracted resume data and recruiter notes.",
        key=f"confirm_delete_{selected_delete_record['id']}"
    )

    if st.button(
        "Permanently delete candidate",
        type="secondary",
        use_container_width=True,
        disabled=not confirm_delete,
        key=f"delete_button_{selected_delete_record['id']}"
    ):
        deleted_hash = selected_delete_record["sha256"]
        deleted_id = selected_delete_record["id"]
        deleted_name = selected_delete_record["name"]

        was_deleted = delete_candidate(deleted_id)

        if was_deleted:
            # Stop an uploaded copy from being automatically
            # saved again during the current Streamlit session.
            st.session_state["deleted_hashes"].add(deleted_hash)

            # Clear old analysis results that might contain
            # the deleted candidate.
            st.session_state.pop("results", None)
            st.session_state.pop("job", None)
            st.session_state.pop("analysis_mode", None)

            st.success(
                f"{deleted_name} has been deleted from "
                "the local candidate database."
            )

            st.rerun()

        else:
            st.error(
                "The candidate record could not be found. "
                "Refresh the page and try again."
            )

else:
    st.info(
        "No saved candidate records match this search."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    f"Resume Intelligence · HR Recruitment Dashboard · "
    f"Version {APP_VERSION} · Semantic matching with human review"
)
