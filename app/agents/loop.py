# app/agent/loop.py
import json
from typing import AsyncGenerator
from google import genai
from google.genai import types

from app.agents.tools import REGISTERED_TOOLS, TOOL_MAP

client = genai.Client()


async def run_agent_loop(
    prompt: str, system_instruction: str, max_turns: int = 5
) -> AsyncGenerator[str, None]:
    """
    Independent Agent Execution Loop.
    Handles tool discovery, execution, and multi-turn response synthesis.
    """
    messages = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

    for turn in range(max_turns):
        # 1. Call Gemini with available tools
        response = await client.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=REGISTERED_TOOLS,
                temperature=0.1,
            ),
        )

        # 2. Append model response to conversation history
        if response.candidates:
            messages.append(response.candidates[0].content)

        # 3. Base Case: If no tools called, stream text and break
        if not response.function_calls:
            yield f"data: {json.dumps({'token': response.text})}\n\n"
            break

        # 4. Action Case: Execute requested tool(s)
        for call in response.function_calls:
            yield f"data: {json.dumps({'status': f'Executing {call.name}', 'args': call.args})}\n\n"

            tool_func = TOOL_MAP.get(call.name)
            if tool_func:
                tool_result = await tool_func(**call.args)
            else:
                tool_result = {"error": f"Tool '{call.name}' not found."}

            yield f"data: {json.dumps({'status': f'Finished {call.name}', 'result': tool_result})}\n\n"

            # 5. Send tool execution result back into conversation history
            messages.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=call.name, response={"result": tool_result}
                        )
                    ],
                )
            )

    yield "data: [DONE]\n\n"
