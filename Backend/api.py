import os
import sqlite3
import json
import time
import asyncio
from typing import Dict, List
from fastapi import FastAPI, Request, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import math
import numpy as np

# =========================
# CONFIG
# =========================
DB_PATH = "data/medic.db"
app = FastAPI(title="MEDIC Backend", version="2.2 (auto resource restore)")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SSE Clients ----------
clients: List[asyncio.Queue] = []


async def push_event(payload: dict):
    """Broadcast an event to all SSE clients."""
    dead = []
    for q in clients:
        try:
            await q.put(payload)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            clients.remove(q)
        except ValueError:
            pass


# ---------- Database Helpers ----------
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_resources_dict(conn) -> Dict[str, int]:
    cur = conn.cursor()
    cur.execute("SELECT type, available FROM resources")
    return {r["type"]: r["available"] for r in cur.fetchall()}


def update_resources_after_delete(conn, triage: str):
    """Restore resources after patient deletion based on triage."""
    triage_map = {
        "Critical": {"icu_beds": 1, "oxygen_cylinders": 2, "doctors": 1},
        "Moderate": {"icu_beds": 1, "oxygen_cylinders": 1, "doctors": 1},
        "Stable": {"icu_beds": 0, "oxygen_cylinders": 0, "doctors": 0},
    }
    released = triage_map.get(triage, {})
    cur = conn.cursor()
    for rtype, qty in released.items():
        cur.execute("UPDATE resources SET available = available + ? WHERE type = ?", (qty, rtype))
    conn.commit()


# ---------- Models ----------
class PatientData(BaseModel):
    name: str
    age: int
    blood_pressure: float
    sugar_level: float
    cholesterol: float
    symptoms: str = ""


# ---------- Score Calculation ----------
def compute_seriousness_score(data: PatientData) -> float:
    return float(
        (data.blood_pressure / 120.0)
        + (data.sugar_level / 100.0)
        + (data.cholesterol / 200.0)
    )


def map_score_to_triage(score: float) -> str:
    if score > 3.0:
        return "Critical"
    elif score > 2.0:
        return "Moderate"
    else:
        return "Stable"


def score_confidence(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-(score - 2.5)))


# ---------- Resource Map ----------
TRIAGE_RESOURCE_MAP = {
    "Critical": {"icu_beds": 1, "oxygen_cylinders": 2, "doctors": 1},
    "Moderate": {"icu_beds": 1, "oxygen_cylinders": 1, "doctors": 1},
    "Stable": {"icu_beds": 0, "oxygen_cylinders": 0, "doctors": 0},
}


# ---------- Startup ----------
@app.on_event("startup")
def startup_tasks():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        score REAL,
        triage TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS resources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT UNIQUE,
        available INTEGER,
        shift TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        resources_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()

    # Default resources
    cur.execute("SELECT COUNT(*) as c FROM resources")
    if cur.fetchone()["c"] == 0:
        defaults = [
            ("icu_beds", 10, "day"),
            ("oxygen_cylinders", 50, "day"),
            ("doctors", 10, "day"),
        ]
        cur.executemany(
            "INSERT INTO resources (type, available, shift) VALUES (?, ?, ?)", defaults
        )
        conn.commit()
    conn.close()


# ---------- Routes ----------
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend running"}


@app.get("/api/resources")
async def api_get_resources():
    conn = get_conn()
    res = get_resources_dict(conn)
    conn.close()
    return {"resources": res}


@app.get("/api/bookings")
async def api_get_bookings(limit: int = 20):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT b.id, b.patient_id, b.resources_json, b.created_at, p.name, p.triage, p.score
    FROM bookings b
    JOIN patients p ON p.id = b.patient_id
    ORDER BY b.created_at DESC
    LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    bookings = [
        {
            "booking_id": r["id"],
            "patient_id": r["patient_id"],
            "name": r["name"],
            "triage": r["triage"],
            "score": r["score"],
            "resources": json.loads(r["resources_json"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return {"bookings": bookings}


@app.post("/api/book")
async def api_book_patient(data: PatientData):
    score = compute_seriousness_score(data)
    triage = map_score_to_triage(score)
    confidence = score_confidence(score)
    required = TRIAGE_RESOURCE_MAP.get(triage, {})

    conn = get_conn()
    cur = conn.cursor()
    cur_resources = get_resources_dict(conn)

    # Check resources
    missing = {}
    for rtype, needed in required.items():
        if cur_resources.get(rtype, 0) < needed:
            missing[rtype] = {"needed": needed, "available": cur_resources.get(rtype, 0)}
    if missing:
        conn.close()
        return JSONResponse(status_code=400, content={
            "message": "Insufficient resources",
            "missing": missing,
            "current_resources": cur_resources
        })

    # Deduct and save
    for rtype, needed in required.items():
        cur.execute("UPDATE resources SET available = available - ? WHERE type = ?", (needed, rtype))

    cur.execute("INSERT INTO patients (name, age, score, triage) VALUES (?, ?, ?, ?)",
                (data.name, data.age, score, triage))
    pid = cur.lastrowid
    cur.execute("INSERT INTO bookings (patient_id, resources_json) VALUES (?, ?)",
                (pid, json.dumps(required)))
    conn.commit()

    updated_res = get_resources_dict(conn)
    conn.close()

    # Push realtime event
    asyncio.create_task(push_event({
        "type": "booking_created",
        "patient_id": pid,
        "name": data.name,
        "triage": triage,
        "score": round(score, 3),
        "resources": updated_res,
        "timestamp": int(time.time())
    }))

    return {
        "message": f"Patient {data.name} booked successfully",
        "patient_id": pid,
        "triage": triage,
        "score": round(score, 3),
        "confidence": round(confidence, 3),
        "resources": updated_res
    }


# ---------- Explain All (Simulated SHAP) ----------
@app.get("/xai/explain_all")
async def explain_all_patients():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, score FROM patients ORDER BY created_at DESC LIMIT 20")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        shap_list = [0.05 + 0.02 * i for i in range(5)]
        return {"shap_values": shap_list, "recommended_patient": 0}

    scores = np.array([r["score"] or 0 for r in rows], dtype=float)
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    shap_list = (0.05 + 0.05 * norm).round(3).tolist()
    recommended = int(np.argmax(shap_list))
    return {"shap_values": shap_list, "recommended_patient": recommended}


@app.delete("/api/patient/{patient_id}")
async def delete_patient(patient_id: int = Path(..., description="ID of patient to delete")):
    """
    Delete a patient and their bookings from the database.
    Restore the consumed resources automatically.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Check if patient exists
    cur.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = cur.fetchone()
    if not patient:
        conn.close()
        return {"message": f"No patient found with ID {patient_id}"}

    # Fetch booking to know what resources were used
    cur.execute("SELECT resources_json FROM bookings WHERE patient_id = ?", (patient_id,))
    booking = cur.fetchone()
    restored_resources = {}
    if booking and booking["resources_json"]:
        try:
            used_resources = json.loads(booking["resources_json"])
            for rtype, used in used_resources.items():
                if used > 0:
                    cur.execute("SELECT available FROM resources WHERE type = ?", (rtype,))
                    row = cur.fetchone()
                    if row:
                        new_val = row["available"] + used
                        cur.execute(
                            "UPDATE resources SET available = ? WHERE type = ?",
                            (new_val, rtype),
                        )
                        restored_resources[rtype] = new_val
        except Exception as e:
            print("Error restoring resources:", e)

    # Delete the booking and patient
    cur.execute("DELETE FROM bookings WHERE patient_id = ?", (patient_id,))
    cur.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()

    # Get updated resources
    updated_resources = get_resources_dict(conn)
    conn.close()

    # Broadcast real-time update
    payload = {
        "type": "patient_deleted",
        "patient_id": patient_id,
        "resources": updated_resources,
    }
    asyncio.create_task(push_event(payload))

    return {
        "message": f"Patient {patient_id} deleted successfully. Resources restored.",
        "resources": updated_resources,
    }


# ---------- SSE ----------
@app.get("/events")
async def events(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    clients.append(q)

    async def event_stream():
        conn = get_conn()
        init = {"type": "init", "resources": get_resources_dict(conn)}
        conn.close()
        yield f"data: {json.dumps(init)}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield "data: {\"type\": \"ping\"}\n\n"
        finally:
            if q in clients:
                clients.remove(q)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
@app.post("/admin/reset_ids")
async def reset_ids():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('patients', 'bookings')")
    conn.commit()
    conn.close()
    return {"message": "Patient and booking IDs reset successfully"}
