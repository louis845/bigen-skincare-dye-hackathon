import base64
import io
import json
import logging
import os
import re
from typing import AsyncGenerator

import google.generativeai as genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types as genai_types
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_FASHION = """
You are a cutting-edge AI fashion stylist. Your task is to analyze the user's full-body image and provide personalized outfit recommendations.

Your Process:
1.  **Analyze the image** to determine the user's apparent body type and note any existing style elements.
2.  **Consider the user's request** for context (e.g., "suggest a summer outfit," "what to wear to a wedding?").
3.  **Generate 2-3 distinct outfit suggestions.**
4.  **For EACH suggestion, provide**:
    a.  An `outfit_name` (e.g., "Chic Summer Brunch," "Professional Power-Look").
    b.  A `description` of the outfit and why it's a good choice.
    c.  A list of clothing `items` in an array (e.g., ["Linen Button-Down Shirt", "Tailored Chinos", "Leather Loafers"]).
    d.  The `suitable_occasions` for this outfit.
5.  **Provide a general fashion tip.**

Output Format:
Respond ONLY with a single, well-formed JSON object. Do not include any text outside of the JSON.
{{
  "body_type_analysis": "e.g., The user appears to have a pear-shaped body type, with wider hips and a defined waist.",
  "style_suggestions": [
    {{
      "outfit_name": "e.g., Effortless Elegance",
      "description": "e.g., This look balances proportions by drawing attention to the upper body while creating a flattering silhouette.",
      "items": ["A-Line Skirt", "Fitted Bodysuit", "Heeled Sandals", "Crossbody Bag"],
      "suitable_occasions": "e.g., Date night, semi-formal events, dinner with friends."
    }}
  ],
  "general_tip": "e.g., Investing in a good tailor can make even affordable clothing look high-end and perfectly fitted to your body."
}}
"""

class FashionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="fashion_agent")

    async def _repair_json_with_llm(self, broken_json_string: str) -> str:
        # ... (implementation is the same as other agents, omitted for brevity)
        pass

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        llm = genai.GenerativeModel('gemini-2.5-pro')
        repair_llm = genai.GenerativeModel('gemini-2.5-pro')
        image_base64 = ctx.session.state.get("image_base64")
        user_message = ctx.session.state.get("user_message", "Suggest an outfit for me.")

        if not image_base64:
            content = genai_types.Content(role="model", parts=[genai_types.Part(text="I need a full-body image to suggest an outfit.")])
            yield Event(author=self.name, content=content)
            return
        
        pil_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        
        prompt = f"User request: {user_message}"
        response = await llm.generate_content_async(
            [SYSTEM_PROMPT_FASHION, pil_image, prompt],
            generation_config={"response_mime_type": "application/json"}
        )
        
        response_text = response.text
        data = {}
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("FashionAgent: Initial JSON parsing failed. Attempting to repair.")
            repaired_json_str = await self._repair_json_with_llm(response_text, repair_llm)
            try:
                data = json.loads(repaired_json_str)
            except json.JSONDecodeError as e:
                error_msg = f"FashionAgent: Failed to decode JSON even after repair: {e}"
                content = genai_types.Content(role="model", parts=[genai_types.Part(text=error_msg)])
                yield Event(author=self.name, content=content)
                return

        # --- Format JSON into Markdown Tables ---
        markdown_output = f"##  Body Type Analysis\n{data.get('body_type_analysis', 'N/A')}\n\n---\n"
        markdown_output += "##  Outfit Recommendations\n"
        suggestions = data.get('style_suggestions', [])
        
        if not suggestions:
            markdown_output += "Could not generate any specific outfit suggestions based on the image."
        else:
            for sug in suggestions:
                markdown_output += f"### {sug.get('outfit_name', 'Outfit Suggestion')}\n"
                markdown_output += f"_{sug.get('description', '')}_\n\n"
                markdown_output += "| Component | Details |\n"
                markdown_output += "|---|---|\n"
                markdown_output += f"| **Key Items** | {', '.join(sug.get('items', ['N/A']))} |\n"
                markdown_output += f"| **Suitable For** | {sug.get('suitable_occasions', 'N/A')} |\n"
                markdown_output += "\n---\n"
        
        tip = data.get('general_tip', '')
        if tip:
            markdown_output += f"\n** Fashion Tip:** {tip}\n"

        content = genai_types.Content(role="model", parts=[genai_types.Part(text=markdown_output)])
        yield Event(author=self.name, content=content)


async def _repair_json_with_llm(self, broken_json_string: str, repair_llm) -> str:
    """Uses a second LLM call to repair a broken JSON string."""
    repair_prompt = f"""The following text is supposed to be a single JSON object, but it is malformed. Please correct any syntax errors and return ONLY the valid JSON object.

Broken JSON:
```json
{broken_json_string}
```"""
    response = await repair_llm.generate_content_async(
        repair_prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    return response.text

# Add the repair method to the class
FashionAgent._repair_json_with_llm = _repair_json_with_llm 