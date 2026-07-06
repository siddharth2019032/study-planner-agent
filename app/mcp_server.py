import json
import os
from typing import Any
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Academic Assistant MCP Server")

DB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "study_sessions.json"))

def _load_data() -> dict:
    if not os.path.exists(DB_FILE):
        # Create default database structure
        default_data = {
            "sessions": [
                {"subject": "Algorithms", "duration_minutes": 90, "date": "2026-07-04", "notes": "Studied dynamic programming"},
                {"subject": "Chemistry", "duration_minutes": 60, "date": "2026-07-05", "notes": "Prepared for organic chem quiz"}
            ],
            "deadlines": [
                {"subject": "Algorithms", "assignment": "Problem Set 4", "deadline": "2026-07-10"},
                {"subject": "Chemistry", "assignment": "Lab Report 3", "deadline": "2026-07-12"},
                {"subject": "Physics", "assignment": "Midterm Exam", "deadline": "2026-07-15"}
            ],
            "goals": [
                {"subject": "Algorithms", "goal": "Master graph traversal and DP"},
                {"subject": "Chemistry", "goal": "Maintain an A grade"},
                {"subject": "Physics", "goal": "Solve all practice midterm exams"}
            ]
        }
        with open(DB_FILE, "w") as f:
            json.dump(default_data, f, indent=2)
        return default_data

    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"sessions": [], "deadlines": [], "goals": []}

def _save_data(data: dict):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)


@mcp.tool()
def get_calendar_events() -> str:
    """Retrieves upcoming calendar classes and events for the student."""
    events = [
        "Monday: 10:00 AM - Algorithms Lecture",
        "Monday: 02:00 PM - Chemistry Lab",
        "Wednesday: 10:00 AM - Algorithms Lecture",
        "Wednesday: 04:00 PM - Study Group Meeting",
        "Friday: 10:00 AM - Physics Recitation"
    ]
    return "\n".join(events)


@mcp.tool()
def get_assignment_deadlines() -> str:
    """Retrieves the list of upcoming assignments, projects, and exam deadlines."""
    data = _load_data()
    deadlines = data.get("deadlines", [])
    if not deadlines:
        return "No upcoming deadlines found."
    lines = []
    for d in deadlines:
        lines.append(f"- [{d['subject']}] {d['assignment']} (Due: {d['deadline']})")
    return "\n".join(lines)


@mcp.tool()
def save_study_session(subject: str, duration_minutes: int, notes: str = "") -> str:
    """Logs a completed study session for tracking academic progress.
    
    Args:
        subject: The subject studied.
        duration_minutes: The duration of the study session in minutes.
        notes: Brief notes about what was studied.
    """
    import datetime
    data = _load_data()
    new_session = {
        "subject": subject,
        "duration_minutes": duration_minutes,
        "date": datetime.date.today().isoformat(),
        "notes": notes
    }
    data["sessions"].append(new_session)
    _save_data(data)
    return f"Successfully logged study session: {duration_minutes} minutes of {subject} on {new_session['date']}."


@mcp.tool()
def get_personal_learning_goals() -> str:
    """Retrieves the student's personal learning goals and study targets."""
    data = _load_data()
    goals = data.get("goals", [])
    if not goals:
        return "No personal learning goals found."
    lines = []
    for g in goals:
        lines.append(f"- {g['subject']}: {g['goal']}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
