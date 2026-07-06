# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


def test_agent_stream() -> None:
    """
    Integration test for the agent stream functionality.
    Tests that the agent returns valid streaming responses.
    """

    session_service = InMemorySessionService()

    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    message = types.Content(
        role="user", parts=[types.Part.from_text(text="Why is the sky blue?")]
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one message"

    has_text_content = False
    for event in events:
        if (
            event.content
            and event.content.parts
            and any(part.text for part in event.content.parts)
        ):
            has_text_content = True
            break
    assert has_text_content, "Expected at least one message with text content"


def test_workflow_approval_resumption() -> None:
    """
    Integration test for the workflow approval and resumption flow.
    Tests that a study plan request interrupts the workflow, and
    providing 'approve' resumes the workflow and yields the confirmation event.
    """
    import unittest.mock
    from google.adk.models.google_llm import Gemini
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    mock_candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Here is a draft of your daily Math study plan, focusing on using your 2 hours effectively:\n\n**Math Study Plan (2 Hours Daily)**\n\n* Daily Slot: 6:00 PM - 8:00 PM")]
        )
    )
    mock_generate_content_response = types.GenerateContentResponse(
        candidates=[mock_candidate]
    )
    mock_llm_response = LlmResponse.create(mock_generate_content_response)

    async def mock_generate_content_async(self, llm_request, stream=False):
        yield mock_llm_response

    from app.agent import root_agent
    session_service = InMemorySessionService()

    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    # Patch the Gemini model call during the test run
    with unittest.mock.patch.object(Gemini, "generate_content_async", mock_generate_content_async):
        # Turn 1: Request a study plan
        message = types.Content(
            role="user", parts=[types.Part.from_text(text="I want a study plan for Math. I have 2 hours daily.")]
        )

        events_t1 = list(
            runner.run(
                new_message=message,
                user_id="test_user",
                session_id=session.id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

        # Find the interrupt event
        interrupt_id = None
        for e in events_t1:
            if e.long_running_tool_ids:
                interrupt_id = list(e.long_running_tool_ids)[0]
                break

        assert interrupt_id == "approve_response", "Expected to be interrupted with 'approve_response'"

        # Turn 2: Approve the plan
        approve_message = types.Content(
            role="user", parts=[types.Part.from_text(text="approve")]
        )

        events_t2 = list(
            runner.run(
                new_message=approve_message,
                user_id="test_user",
                session_id=session.id,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        )

    # Check that we received the approval confirmation event and completed
    has_approval_event = any(
        e.content and any(
            "Study plan approved and locked in" in part.text for part in e.content.parts if part.text
        )
        for e in events_t2
    )
    assert has_approval_event, "Expected a plan_reviewer event confirming approval"

    has_final_output = any(
        e.content and any(
            "Original Plan" in part.text for part in e.content.parts if part.text
        )
        for e in events_t2
    )
    assert has_final_output, "Expected a final_output event containing the approved plan"



