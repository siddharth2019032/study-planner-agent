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

from typing import Any, Optional, AsyncGenerator
from google.adk.events.event import Event
from google.adk.agents.run_config import RunConfig
from google.genai import types as genai_types
from google.adk.runners import Runner

original_run_async = Runner.run_async

async def patched_run_async(
    self,
    *,
    user_id: str,
    session_id: str,
    invocation_id: Optional[str] = None,
    new_message: Optional[genai_types.Content] = None,
    state_delta: Optional[dict[str, Any]] = None,
    run_config: Optional[RunConfig] = None,
    yield_user_message: bool = False,
) -> AsyncGenerator[Event, None]:
    try:
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session and session.events:
            unresolved_interrupt_ids = set()
            interrupt_event_map = {}
            for e in session.events:
                if e.author != "user" and getattr(e, "long_running_tool_ids", None):
                    for tid in e.long_running_tool_ids:
                        unresolved_interrupt_ids.add(tid)
                        interrupt_event_map[tid] = e
                elif e.author == "user" and e.content and e.content.parts:
                    for p in e.content.parts:
                        if p.function_response and p.function_response.id:
                            unresolved_interrupt_ids.discard(p.function_response.id)
            
            if unresolved_interrupt_ids and new_message and new_message.parts:
                interrupt_id = list(unresolved_interrupt_ids)[0]
                interrupt_event = interrupt_event_map[interrupt_id]
                has_text = any(p.text for p in new_message.parts)
                has_fr = any(p.function_response for p in new_message.parts)
                if has_text and not has_fr:
                    user_text = "".join(p.text for p in new_message.parts if p.text)
                    
                    fr_part = genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            id=interrupt_id,
                            name="adk_request_input",
                            response={"result": user_text}
                        )
                    )
                    new_message = genai_types.Content(
                        role="user",
                        parts=[fr_part]
                    )
                    invocation_id = interrupt_event.invocation_id
    except Exception as e:
        import logging
        logging.error(f"Error in auto-resume check: {e}", exc_info=True)

    async for event in original_run_async(
        self,
        user_id=user_id,
        session_id=session_id,
        invocation_id=invocation_id,
        new_message=new_message,
        state_delta=state_delta,
        run_config=run_config,
        yield_user_message=yield_user_message,
    ):
        yield event

Runner.run_async = patched_run_async

from .agent import app

__all__ = ["app"]
