import os
import re
import datetime
from typing import Any, AsyncGenerator, Generator
from pydantic import BaseModel, Field

from google.adk.workflow import Workflow, START, node, DEFAULT_ROUTE, Edge, FunctionNode
from google.adk.agents import LlmAgent
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.tools import AgentTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.apps import App, ResumabilityConfig
from google.genai import types

from app.config import config

# --- Structured Output Schemas ---

class StudySession(BaseModel):
    subject: str = Field(description="Subject name")
    time: str = Field(description="Time slot (e.g. 10:00 AM - 11:30 AM)")
    task: str = Field(description="What specific topic/task to study or revise")
    priority: str = Field(description="Priority: High, Medium, or Low")

class StudyPlan(BaseModel):
    timetable: list[StudySession] = Field(description="List of daily or weekly study sessions")
    tips: list[str] = Field(description="Personalized study tips based on the schedule")

class QuizQuestion(BaseModel):
    question: str = Field(description="The question prompt")
    options: list[str] = Field(description="Four multiple-choice options")
    correct_answer: str = Field(description="The exact correct option matching one of the options")
    explanation: str = Field(description="Explanation of why this answer is correct")

class Quiz(BaseModel):
    subject: str = Field(description="The subject of this quiz")
    questions: list[QuizQuestion] = Field(description="List of quiz questions")

class SubjectProgress(BaseModel):
    subject: str = Field(description="Subject name")
    sessions_completed: int = Field(description="Number of study sessions completed")
    target_hours: float = Field(description="Target study hours")
    status: str = Field(description="Current status (e.g. Behind, On Track, Ahead)")

class ProgressReport(BaseModel):
    overall_summary: str = Field(description="Summary of overall study progress")
    subjects: list[SubjectProgress] = Field(description="Progress per subject")
    motivation_tip: str = Field(description="Personalized motivational tip")

class CoachFeedback(BaseModel):
    coaching_tips: list[str] = Field(description="Actionable study tips/strategies")
    motivation_quote: str = Field(description="Inspirational advice or quote")
    next_steps: list[str] = Field(description="Recommended next actions")


# --- MCP Toolset Configuration ---

mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "python", "-m", "app.mcp_server"]
        )
    )
)


# --- Specialized LlmAgent Sub-Agents ---

planner_agent = LlmAgent(
    name="planner_agent",
    model=config.model,
    instruction="""You are a professional study scheduler.
Create a structured daily/weekly study timetable, prioritized task list, and revision plan based on the student's available hours, exam dates, deadlines, and goals.
Retrieve any student academic events and assignment deadlines using your MCP tools before creating the plan.
IMPORTANT: Your output MUST be a valid JSON object matching the StudyPlan schema. Do not include any introductory or conversational text, markdown wrapping (such as ```json), or explanation. Only return the valid JSON data structure.""",
    output_schema=StudyPlan,
    tools=[mcp_toolset],
    description="Generates personalized study timetables, revision plans, and task lists."
)

study_coach_agent = LlmAgent(
    name="study_coach_agent",
    model=config.model,
    instruction="""You are an academic coach and mentor.
Provide students with effective study techniques (like active recall, spaced repetition), memory advice, and motivational tips.
Offer strategies to avoid burnout.""",
    output_schema=CoachFeedback,
    description="Provides academic coaching, motivational advice, and personalized study tips."
)

quiz_generator_agent = LlmAgent(
    name="quiz_generator_agent",
    model=config.model,
    instruction="""You are an interactive tutor.
Generate quiz questions (multiple-choice or short answer) based on the student's subjects and topics.
Always explain the answers to help the student learn.""",
    output_schema=Quiz,
    description="Generates academic quizzes and explanations for student self-testing."
)

progress_tracker_agent = LlmAgent(
    name="progress_tracker_agent",
    model=config.model,
    instruction="""You are a progress tracking assistant.
Help students log completed sessions, compare actual hours studied vs targets, and monitor overall subject mastery.
Use your MCP tools to log completed study sessions or fetch student learning goals.""",
    output_schema=ProgressReport,
    tools=[mcp_toolset],
    description="Tracks study achievements and maintains a log of student progress."
)


# --- Orchestrator LlmAgent ---

orchestrator_agent = LlmAgent(
    name="academic_orchestrator",
    model=config.model,
    instruction="""You are the Academic Coordinator and main interface for the student.
Your role is to understand the student's request and delegate tasks to the appropriate specialized assistants:
- Use planner_agent to create study plans, schedules, and revision timetables.
- Use study_coach_agent to get study tips, motivation, and learning advice.
- Use quiz_generator_agent to generate learning quizzes.
- Use progress_tracker_agent to log and review academic progress.

If the student is requesting a study plan, draft one and tell them it will require their approval before finalization.
Always respond to the student politely and present the answers from the sub-agents clearly.
""",
    tools=[
        AgentTool(planner_agent),
        AgentTool(study_coach_agent),
        AgentTool(quiz_generator_agent),
        AgentTool(progress_tracker_agent)
    ],
    output_key="orchestrator_response"
)


# --- Workflow Graph Nodes ---

def security_checkpoint(ctx: Context, node_input: types.Content) -> Event:
    text = ""
    if node_input and node_input.parts:
        text = "".join(part.text for part in node_input.parts if part.text)
    
    # 1. Prompt injection detection
    injection_keywords = ["ignore previous instructions", "system prompt", "bypass guardrails", "override rules"]
    is_injection = False
    if config.injection_detection_enabled:
        for kw in injection_keywords:
            if kw in text.lower():
                is_injection = True
                break
                
    if is_injection:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "PROMPT_INJECTION_DETECTED",
            "severity": "CRITICAL",
            "input_preview": text[:100]
        }
        print(f"AUDIT_LOG: {log_entry}")
        return Event(
            output="Security check failed: potential prompt injection detected.",
            route="SECURITY_EVENT"
        )

    # 2. Domain-Specific Rule: Academic Dishonesty Detection
    dishonesty_keywords = ["cheat in exam", "plagiarize", "buy essay", "bypass turnitin", "cheat on exam", "hack grades"]
    is_dishonest = False
    for kw in dishonesty_keywords:
        if kw in text.lower():
            is_dishonest = True
            break
            
    if is_dishonest:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "ACADEMIC_DISHONESTY_DETECTED",
            "severity": "WARNING",
            "input_preview": text[:100]
        }
        print(f"AUDIT_LOG: {log_entry}")
        return Event(
            output="Security check failed: Academic dishonesty or cheating request detected.",
            route="SECURITY_EVENT"
        )

    # 3. PII Scrubbing (regex for student email and phone numbers)
    scrubbed_text = text
    if config.pii_redaction_enabled:
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        phone_pattern = r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        scrubbed_text = re.sub(email_pattern, "[EMAIL_REDACTED]", scrubbed_text)
        scrubbed_text = re.sub(phone_pattern, "[PHONE_REDACTED]", scrubbed_text)
        
        if scrubbed_text != text:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "event": "PII_REDACTED",
                "severity": "WARNING",
                "input_preview": scrubbed_text[:100]
            }
            print(f"AUDIT_LOG: {log_entry}")

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": "SECURITY_PASS",
        "severity": "INFO"
    }
    print(f"AUDIT_LOG: {log_entry}")

    scrubbed_content = types.Content(role="user", parts=[types.Part.from_text(text=scrubbed_text)])
    
    return Event(
        output=scrubbed_content,
        route="__DEFAULT__",
        state={"clean_input": scrubbed_text}
    )


def security_alert(node_input: str) -> Event:
    content = types.Content(
        role="model",
        parts=[types.Part.from_text(text=f"⚠️ Security Warning: {node_input}")]
    )
    return Event(output=node_input, content=content)


async def plan_reviewer(ctx: Context, node_input: Any) -> AsyncGenerator[Event, None]:
    text = ""
    if isinstance(node_input, str):
        text = node_input
    elif node_input and hasattr(node_input, "parts"):
        text = "".join(part.text for part in node_input.parts if part.text)
    else:
        text = str(node_input)
        
    clean_input = ctx.state.get("clean_input", "")
    is_plan_request = any(w in clean_input.lower() for w in ["plan", "schedule", "timetable"])
    plan_approved = ctx.state.get("plan_approved", False)
    
    if is_plan_request and not plan_approved:
        if not ctx.resume_inputs or "approve_response" not in ctx.resume_inputs:
            yield RequestInput(
                interrupt_id="approve_response",
                message="✋ Please review the study plan draft above. Type 'approve' to confirm, or describe any changes you want to make:"
            )
            return
        
        user_response = ctx.resume_inputs["approve_response"]
        if isinstance(user_response, dict):
            user_response = user_response.get("result", "") or user_response.get("response", "")
        if not isinstance(user_response, str):
            user_response = str(user_response)

        if "approve" in user_response.lower():
            yield Event(
                output=f"Study plan approved!\n\nOriginal Plan:\n{text}",
                content=types.Content(role="model", parts=[types.Part.from_text(text="✅ Study plan approved and locked in! Good luck with your studies!")]),
                state={"plan_approved": True},
                route="__DEFAULT__"
            )
        else:
            revision_input = f"The student wants to modify the plan. Feedback: {user_response}"
            yield Event(
                output=types.Content(role="user", parts=[types.Part.from_text(text=revision_input)]),
                content=types.Content(role="model", parts=[types.Part.from_text(text=f"🔄 Sending request back to coordinator for adjustments: '{user_response}'")]),
                state={"clean_input": revision_input},
                route="revise"
            )
    else:
        yield Event(output=text, route="__DEFAULT__")


plan_reviewer_node = FunctionNode(
    func=plan_reviewer,
    name="plan_reviewer",
    rerun_on_resume=True
)


def final_output(node_input: Any) -> Event:
    text = str(node_input)
    content = types.Content(
        role="model",
        parts=[types.Part.from_text(text=text)]
    )
    return Event(output=text, content=content)


# --- Workflow Definition ---

root_agent = Workflow(
    name="academic_coordinator",
    edges=[
        ('START', security_checkpoint),
        (security_checkpoint, {
            "SECURITY_EVENT": security_alert,
            "__DEFAULT__": orchestrator_agent,
        }),
        (orchestrator_agent, plan_reviewer_node),
        (plan_reviewer_node, {
            "revise": orchestrator_agent,
            "__DEFAULT__": final_output,
        }),
        (security_alert, final_output)
    ],
    description="A multi-agent workflow for student academic planning, quiz generation, and coaching."
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True)
)
