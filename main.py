import os
import re
import uuid
from datetime import datetime
from typing import List, Optional

import dateparser
import psycopg2
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
from transformers import pipeline


# --------- Config ---------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/meetingmind",
)

MAX_SUMMARY_INPUT_CHARS = 3500
MAX_QA_CONTEXT_CHARS = 4000


# --------- App ---------
app = FastAPI(title="MeetingMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Load NLP models once at startup ---------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model missing. Run: python -m spacy download en_core_web_sm"
    )

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


# --------- Request/Response models ---------
class AnalyzeRequest(BaseModel):
    title: str
    transcript: str


class AskRequest(BaseModel):
    meeting_id: str
    question: str


# --------- DB helpers ---------
def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    ddl = """
    CREATE TABLE IF NOT EXISTS meetings (
        id UUID PRIMARY KEY,
        title TEXT NOT NULL,
        transcript TEXT NOT NULL,
        summary TEXT NOT NULL,
        followup_suggestions JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS action_items (
        id UUID PRIMARY KEY,
        meeting_id UUID NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
        task_text TEXT NOT NULL,
        owner TEXT,
        deadline_raw TEXT,
        deadline_iso TEXT
    );

    CREATE TABLE IF NOT EXISTS qa_history (
        id UUID PRIMARY KEY,
        meeting_id UUID NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        score FLOAT,
        asked_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


@app.on_event("startup")
def on_startup():
    init_db()


# --------- NLP helpers ---------
def preprocess_transcript(text: str) -> str:
    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_summary(text: str) -> str:
    short_text = text[:MAX_SUMMARY_INPUT_CHARS]
    result = summarizer(
        short_text,
        max_length=160,
        min_length=60,
        do_sample=False,
    )
    return result[0]["summary_text"]


def find_owner(sentence: str, doc) -> Optional[str]:
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    # fallback pattern: "X will ...", "X can ..."
    match = re.match(r"^([A-Z][a-z]+)\s+(will|can|should)\b", sentence)
    if match:
        return match.group(1)
    return None


def normalize_deadline(sentence: str) -> tuple[Optional[str], Optional[str]]:
    date_patterns = [
        r"\bby\s+([A-Za-z0-9,\s]+)",
        r"\bbefore\s+([A-Za-z0-9,\s]+)",
        r"\bon\s+([A-Za-z0-9,\s]+)",
        r"\bnext\s+[A-Za-z]+\b",
        r"\bthis\s+[A-Za-z]+\b",
        r"\btomorrow\b",
        r"\bfriday\b|\bmonday\b|\btuesday\b|\bwednesday\b|\bthursday\b|\bsaturday\b|\bsunday\b",
    ]
    for pattern in date_patterns:
        m = re.search(pattern, sentence, flags=re.IGNORECASE)
        if m:
            raw_date = m.group(1).strip() if m.groups() else m.group(0).strip()
            parsed = dateparser.parse(raw_date)
            iso_date = parsed.date().isoformat() if parsed else None
            return raw_date, iso_date
    return None, None


def extract_action_items(text: str) -> List[dict]:
    doc = nlp(text)
    items: List[dict] = []
    action_triggers = (
        "will",
        "should",
        "need to",
        "must",
        "please",
        "action item",
        "todo",
        "follow up",
    )

    for sent in doc.sents:
        s = sent.text.strip()
        s_lower = s.lower()
        if any(trigger in s_lower for trigger in action_triggers):
            owner = find_owner(s, nlp(s))
            raw_deadline, iso_deadline = normalize_deadline(s)
            items.append(
                {
                    "task_text": s,
                    "owner": owner,
                    "deadline_raw": raw_deadline,
                    "deadline_iso": iso_deadline,
                }
            )
    return items


def suggest_followups(text: str, action_items: List[dict]) -> List[str]:
    suggestions = []
    lower_text = text.lower()

    unresolved_signals = ["blocker", "pending", "not decided", "open issue", "follow up"]
    if any(sig in lower_text for sig in unresolved_signals):
        suggestions.append("Schedule a follow-up meeting to resolve pending blockers.")

    no_deadline_count = sum(1 for i in action_items if not i.get("deadline_iso"))
    if no_deadline_count > 0:
        suggestions.append(
            f"{no_deadline_count} action item(s) have no clear deadline. Plan a short alignment meeting."
        )

    if len(action_items) >= 5:
        suggestions.append("Large number of tasks detected. Add a weekly checkpoint meeting.")

    if not suggestions:
        suggestions.append("No urgent follow-up meeting required based on current transcript.")
    return suggestions


# --------- API routes ---------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    cleaned = preprocess_transcript(req.transcript)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    summary = generate_summary(cleaned)
    action_items = extract_action_items(cleaned)
    followups = suggest_followups(cleaned, action_items)

    meeting_id = str(uuid.uuid4())
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO meetings (id, title, transcript, summary, followup_suggestions)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (meeting_id, req.title, cleaned, summary, str(followups).replace("'", '"')),
            )

            for item in action_items:
                cur.execute(
                    """
                    INSERT INTO action_items (id, meeting_id, task_text, owner, deadline_raw, deadline_iso)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        meeting_id,
                        item["task_text"],
                        item["owner"],
                        item["deadline_raw"],
                        item["deadline_iso"],
                    ),
                )
        conn.commit()

    return {
        "meeting_id": meeting_id,
        "title": req.title,
        "summary": summary,
        "action_items": action_items,
        "followup_suggestions": followups,
    }


@app.post("/ask")
def ask(req: AskRequest):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT transcript FROM meetings WHERE id = %s", (req.meeting_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Meeting not found.")

            transcript = row["transcript"][:MAX_QA_CONTEXT_CHARS]
            qa = qa_model(question=req.question, context=transcript)
            answer = qa["answer"]
            score = float(qa["score"])

            cur.execute(
                """
                INSERT INTO qa_history (id, meeting_id, question, answer, score)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (str(uuid.uuid4()), req.meeting_id, req.question, answer, score),
            )
        conn.commit()

    return {"question": req.question, "answer": answer, "confidence": score}


@app.get("/meeting/{meeting_id}")
def get_meeting(meeting_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, summary, followup_suggestions, created_at
                FROM meetings
                WHERE id = %s
                """,
                (meeting_id,),
            )
            meeting = cur.fetchone()
            if not meeting:
                raise HTTPException(status_code=404, detail="Meeting not found.")

            cur.execute(
                """
                SELECT task_text, owner, deadline_raw, deadline_iso
                FROM action_items
                WHERE meeting_id = %s
                """,
                (meeting_id,),
            )
            actions = cur.fetchall()

    return {"meeting": meeting, "action_items": actions}
