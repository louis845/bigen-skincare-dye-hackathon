import base64
import json
import logging
import mimetypes
import os
import tempfile
from io import BytesIO
from typing import AsyncGenerator

import gradio as gr
from dotenv import load_dotenv
from google.adk.agents import SequentialAgent, LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from PIL import Image

from .fashion_agent import FashionAgent
from .image_handler import pil_image_to_base64
from .skincare_agent import SkincareAgent
# from .haircut_agent import HaircutAgent  # Commented out

  # Load environment variables from .env

# Initialize agents
skincare_agent = SkincareAgent()
fashion_agent = FashionAgent()
# Haircut parts: commented out for now, to be added later
# haircut_agent = HaircutAgent()

# Initialize ADK agents
skincare_adk = SkincareAgent()
fashion_adk = FashionAgent()
# haircut_adk = HaircutAgent()

# Create router agent to determine analysis type
router = LlmAgent(
    name="router",
    model=os.environ['GEMINI_MODEL_NAME'],
    instruction="Based on the query and/or image, determine if it's about skincare (facial focus), fashion (full body), or haircut (hair focus). Respond with ONLY one word: 'skincare', 'fashion', or 'haircut'. Do not add any explanations or extra text.",
    output_key="agent_type"
)

class StylistOrchestrator(BaseAgent):
    def __init__(self, name, sub_agents):
        super().__init__(name=name, sub_agents=sub_agents)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting orchestration.")
        async for event in self.sub_agents[0].run_async(ctx):  # Run router
            yield event
        agent_type = ctx.session.state.get('agent_type')
        logger.info(f"[{self.name}] Router decided: {agent_type}")
        if agent_type == 'skincare':
            logger.info(f"[{self.name}] Running skincare agent.")
            async for event in self.sub_agents[1].run_async(ctx):
                yield event
        elif agent_type == 'fashion':
            logger.info(f"[{self.name}] Running fashion agent.")
            async for event in self.sub_agents[2].run_async(ctx):
                yield event
        elif agent_type == 'haircut':
            # Placeholder for haircut agent (to be added later)
            yield Event(text="Haircut agent not yet implemented.")
        else:
            logger.warning(f"[{self.name}] Unknown agent type: {agent_type}")
            yield Event(text="Unknown analysis type.")

orchestrator = StylistOrchestrator(
    name="stylist_orchestrator",
    sub_agents=[router, skincare_adk, fashion_adk]
)

# Set up ADK session and runner
session_service = InMemorySessionService()
runner = Runner(agent=orchestrator, app_name="ai_stylist", session_service=session_service)

# Remove hard-coded API key
# os.environ["GOOGLE_API_KEY"] = "your-google-api-key"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('google_adk').setLevel(logging.ERROR)
logging.getLogger('google_genai').setLevel(logging.ERROR)

# --- Thinking Indicator ---
thinking_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 30" fill="#9A9A9A">
  <circle cx="15" cy="15" r="15">
    <animate attributeName="r" from="15" to="15" begin="0s" dur="0.8s" values="15;9;15" calcMode="linear" repeatCount="indefinite" />
    <animate attributeName="fill-opacity" from="1" to="1" begin="0s" dur="0.8s" values="1;.5;1" calcMode="linear" repeatCount="indefinite" />
  </circle>
  <circle cx="60" cy="15" r="9" fill-opacity="0.3">
    <animate attributeName="r" from="9" to="9" begin="0s" dur="0.8s" values="9;15;9" calcMode="linear" repeatCount="indefinite" />
    <animate attributeName="fill-opacity" from="0.5" to="0.5" begin="0s" dur="0.8s" values=".5;1;.5" calcMode="linear" repeatCount="indefinite" />
  </circle>
  <circle cx="105" cy="15" r="15">
    <animate attributeName="r" from="15" to="15" begin="0s" dur="0.8s" values="15;9;15" calcMode="linear" repeatCount="indefinite" />
    <animate attributeName="fill-opacity" from="1" to="1" begin="0s" dur="0.8s" values="1;.5;1" calcMode="linear" repeatCount="indefinite" />
  </circle>
</svg>"""
encoded_thinking_svg = base64.b64encode(thinking_svg.encode("utf-8")).decode("utf-8")
thinking_indicator_html = f'<div style="display: flex; align-items: center;"><img src="data:image/svg+xml;base64,{encoded_thinking_svg}" alt="thinking..." style="width: 40px;" /></div>'

def process_image(image, agent_type):
    image_base64 = pil_image_to_base64(image)
    if agent_type == "Skincare":
        result = skincare_agent.analyze_face(image_base64)
    elif agent_type == "Fashion":
        result = fashion_agent.analyze_body(image_base64)
    else:
        return "Invalid agent type"
    return result  # Directly return the text

async def process_image_adk(image, agent_type):
    image_base64 = pil_image_to_base64(image)
    session_id = "session_" + agent_type.lower()
    await session_service.create_session(app_name="ai_stylist", user_id="user", session_id=session_id, state={"image_base64": image_base64})
    content = types.Content(role='user', parts=[types.Part(text=f"Analyze for {agent_type} with image: {image_base64}")])
    response_text = ""
    async for event in runner.run_async(user_id="user", session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            response_text += event.content.parts[0].text + "\n"
    return response_text  # Return accumulated text

def image_to_base64_str(pil_image: Image.Image) -> str:
    """Converts a PIL image to a base64 string for display."""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def audio_to_base64_str(audio_filepath: str) -> str:
    """Converts an audio file to a base64 string for display."""
    with open(audio_filepath, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode()
    return f"data:audio/wav;base64,{encoded_string}"

async def chatbot_handler(message: str, history: list, image: object):
    user_message = message if message else "Analyze this image."
    image_base64 = pil_image_to_base64(image) if image else None
    user_content = user_message
    if image_base64:
        user_content += f'<div><img src="data:image/jpeg;base64,{image_base64}" alt="User uploaded image" style="max-height: 250px;"></div>'
    history.append([user_content, thinking_indicator_html])
    yield history, "", None # Update chatbot, clear inputs
    session_id = "session_dynamic"
    state = {"image_base64": image_base64, "user_message": user_message}
    await session_service.create_session(app_name="ai_stylist", user_id="user", session_id=session_id, state=state)
    parts = [types.Part(text=user_message)]
    if image_base64:
        parts.append(types.Part(inline_data={"mime_type": "image/jpeg", "data": image_base64}))
    content = types.Content(role='user', parts=parts)
    bot_response_text = ""
    first_chunk = True
    async for event in runner.run_async(user_id="user", session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            chunk = event.content.parts[0].text
            if chunk and chunk.strip().lower() not in ['skincare', 'fashion', 'haircut']:
                if first_chunk:
                    bot_response_text = chunk
                    history[-1][1] = bot_response_text
                    first_chunk = False
                else:
                    bot_response_text += chunk
                    history[-1][1] = bot_response_text
                yield history, "", None
    logger.info("Final bot response complete.")

def clear_chat():
    return [], "", None, gr.update(interactive=False), []

# Updated Gradio UI without agent_dropdown
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), title="AI Stylist") as demo:
    gr.Markdown("# �� AI Stylist - AI 造型師")

    chatbot = gr.Chatbot(label="AI Stylist Chat", height=500, avatar_images=(None, "https://i.imgur.com/18wBez3.png"), show_copy_button=True)
    chatbot_state = gr.State([]) # Use messages format

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image", scale=1)
        with gr.Column(scale=3):
            msg = gr.Textbox(label="Your Question", placeholder="Type a question or upload an image...", interactive=True)
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary", interactive=False)
                clear_btn = gr.Button("Clear")

    gr.Examples(
        ["Analyze my skin", "Suggest outfits for this body type"],
        inputs=[msg], label="Example Questions"
    )

    def toggle_submit(msg: str, image: object) -> dict:
        return gr.update(interactive=bool(msg or image))

    def ui_state(is_locked: bool):
        return {
            msg: gr.update(interactive=not is_locked),
            image_input: gr.update(interactive=not is_locked),
            submit_btn: gr.update(visible=not is_locked),
            clear_btn: gr.update(interactive=not is_locked),
        }

    # Event Listeners
    image_input.change(toggle_submit, [msg, image_input], [submit_btn])
    msg.change(toggle_submit, [msg, image_input], [submit_btn])

    submit_btn.click(chatbot_handler, inputs=[msg, chatbot, image_input], outputs=[chatbot, msg, image_input], queue=True)
    clear_btn.click(clear_chat, None, [chatbot, msg, image_input, submit_btn, chatbot_state], queue=False)
