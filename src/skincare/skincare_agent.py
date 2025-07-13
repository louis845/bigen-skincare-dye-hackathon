import base64
import io
import json
import logging
import os
import random
import re
from typing import AsyncGenerator

import google.generativeai as genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types as genai_types
from langchain_core.messages import HumanMessage
from PIL import Image

logger = logging.getLogger(__name__)

# --- Helper Functions & Data ---

def image_to_base64_uri(file_path):
    """Converts a local image file to a base64 data URI."""
    try:
        # Determine mime type from file extension
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime_type = mime_map.get(ext)

        if not mime_type:
            logger.error(f"Unsupported image type for file: {file_path}")
            return ""

        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        logger.error(f"Local product image not found at path: {file_path}")
        return "" # Return empty string if file is not found
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return ""

products = {
    'dry skin': [
        {'name': 'Vaseline Advance Repair Dry Skin Healing Balm 250ml', 'brand': 'Vaseline 凡士林', 'link': 'https://www.mannings.com.hk/en/vaseline-advance-repair-dry-skin-healing-balm-250ml/p/851485', 'image': 'skincare_image/Screenshot 2025-07-12 at 10.31.58 PM.png', 'price': 59.9},
        {'name': 'Cetaphil Gentle Skin Cleanser 500ml + Moisturising Cream 50g', 'brand': 'Cetaphil舒特膚', 'link': 'https://www.mannings.com.hk/en/cetaphil-gentle-skin-cleanser-500ml-moisturising-cream-50g/p/652370', 'image': 'skincare_image/Screenshot 2025-07-12 at 10.35.07 PM.png', 'price': 126}
    ],
    'oily skin': [
        {'name': 'Cetaphil Oily Skin Cleanser 500ml', 'brand': 'Cetaphil舒特膚', 'link': 'https://www.mannings.com.hk/en/cetaphil-oily-skin-cleanser-500ml/p/365502', 'image': 'skincare_image/Screenshot 2025-07-12 at 10.37.25 PM.png', 'price': 150},
        {'name': 'sofina-primavista-long-lasting-primer-for-very-oily-skin-25ml', 'brand': 'Sofina', 'link': 'https://www.mannings.com.hk/en/sofina-primavista-long-lasting-primer-for-very-oily-skin-25ml/p/063016', 'image': 'skincare_image/Screenshot 2025-07-12 at 10.39.18 PM.png', 'price': 240},
        {'name': 'Physiogel Red Soothing Cica Balance Toner 200ml', 'brand': 'Physiogel 潔美淨醫學美肌', 'link': 'https://www.mannings.com.hk/en/physiogel-red-soothing-cica-balance-toner-200ml/p/126532', 'image': 'skincare_image/Screenshot 2025-07-12 at 10.40.31 PM.png', 'price': 179}
    ],
    'acne': [
        {'name': 'Hiruscar Post Acne 10g', 'brand': 'Hiruscar', 'link': 'https://www.mannings.com.hk/en/hiruscar-post-acne-10g/p/125146', 'image': 'skincare_image/Screenshot 2025-07-12 at 10.41.33 PM.png', 'price': 129},
        {'name': 'Dermatix Acne Scar Pro 7g', 'brand': 'Dermatix 倍舒痕', 'link': 'https://www.mannings.com.hk/en/dermatix-acne-scar-pro-7g/p/148478', 'image': 'skincare_image/Screenshot 2025-07-12 at 10.42.29 PM.png', 'price': 145},
        {'name': 'Atorrege AD+ Acne Solution Set (Cool Lotion 150ml + Acne Spot 10ml + Mild Cleansing 40g)', 'brand': 'Atorrege AD+', 'link': 'https://www.mannings.com.hk/en/atorrege-ad-acne-solution-set-cool-lotion-150ml-acne-spot-10ml-mild-cleansing-40g/p/321240', 'image': 'skincare_image/Screenshot 2025-07-12 at 10.46.38 PM.png', 'price': 450}
    ],
    'dark circles': [
        {'name': 'ZINO Dark Circle Removal Golden Eye Mask 30 Pairs', 'brand': 'ZINO', 'link': 'https://www.mannings.com.hk/en/zino-dark-circle-removal-golden-eye-mask-30-pairs/p/125468', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.04.53 PM.png', 'price': 299},
        {'name': "L’Oreal Paris Glycolic Bright Glowing Anti-dark Circles Brightening Eye Serum 20ml", 'brand': "L'Oreal Paris", 'link': 'https://www.mannings.com.hk/en/l-oreal-paris-glycolic-bright-glowing-anti-dark-circles-brightening-eye-serum-20ml/p/795559', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.06.08 PM.png', 'price': 255},
        {'name': 'Nivea Luminous630 Anti Dark-Spot Eye Treatment Cream 15ml', 'brand': 'Nivea', 'link': 'https://www.mannings.com.hk/en/nivea-luminous630-anti-dark-spot-eye-treatment-cream-15ml/p/105148', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.07.13 PM.png', 'price': 205}
    ],
    'wrinkles': [
        {'name': 'Meishoku Wrinkle White Serum 33ml', 'brand': 'Meishoku明色', 'link': 'https://www.mannings.com.hk/en/meishoku-wrinkle-white-serum-33ml/p/715896', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.08.40 PM.png', 'price': 198},
        {'name': 'Kracie Hadabisei Wrinklecare Eyecream 15g', 'brand': 'Kracie肌美精', 'link': 'https://www.mannings.com.hk/en/kracie-hadabisei-wrinklecare-eyecream-15g/p/115519', 'image': None, 'price': 158},
        {'name': 'Kracie Hadabisei Wrinklecare Serum 30ml', 'brand': 'Kracie肌美精', 'link': 'https://www.mannings.com.hk/en/kracie-hadabisei-wrinklecare-serum-30ml/p/115873', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.13.40 PM.png', 'price': 208}
    ],
    'sunburn': [
        {'name': 'Sofina iP Skin Care UV Protect Emulsion (02 For Oily Skin) SPF50+ PA+++ 30ml', 'brand': 'Sofina', 'link': 'https://www.mannings.com.hk/en/sofina-ip-skin-care-uv-protect-emulsion-02-for-oily-skin-spf50-pa-30ml/p/488155', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.15.53 PM.png', 'price': 220},
        {'name': 'Mannings Aloe Vera Gel 350ml', 'brand': 'Mannings萬寧', 'link': 'https://www.mannings.com.hk/en/mannings-aloe-vera-gel-350ml/p/934638', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.16.22 PM.png', 'price': 29.9},
        {'name': 'Nivea Protect & Refresh Sun Lotion SPF50 PA++++ 75ml', 'brand': 'Nivea', 'link': 'https://www.mannings.com.hk/en/nivea-protect-refresh-sun-lotion-spf50-pa-75ml/p/691287', 'image': None, 'price': 68}
    ],
    'face wash': [
        {'name': 'UL.OS Face Wash 100g', 'brand': 'Otsuka', 'link': 'https://www.mannings.com.hk/en/ul-os-face-wash-100g/p/789305', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.17.50 PM.png', 'price': 88},
        {'name': 'Curel Foaming Face Wash 150ml', 'brand': 'Curel', 'link': 'https://www.mannings.com.hk/en/curel-foaming-face-wash-150ml/p/434894', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.18.44 PM.png', 'price': 150},
        {'name': 'Biore Face Wash Mild 100g', 'brand': 'Biore碧柔', 'link': 'https://www.mannings.com.hk/en/biore-face-wash-mild-100g/p/772822', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.19.37 PM.png', 'price': 38.9}
    ],
    'sunscreen': [
        {'name': 'Hawaiian Tropic Dark Tanning Oil Sunscreen Spray SPF4 240ml', 'brand': 'Hawaiian Tropic', 'link': 'https://www.mannings.com.hk/en/hawaiian-tropic-dark-tanning-oil-sunscreen-spray-spf4-240ml/p/341974', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.20.37 PM.png', 'price': 142.9},
        {'name': 'La Roche-Posay Anthelios UVMUNE 400 Oil-Control Fluid 50ml', 'brand': 'La Roche-Posay', 'link': 'https://www.mannings.com.hk/en/la-roche-posay-anthelios-uvmune-400-oil-control-fluid-50ml/p/123877', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.21.35 PM.png', 'price': 265},
        {'name': 'Sofina iP Skin Care UV Protect Emulsion (02 Oil Control) SPF50+ PA+++ 30ml', 'brand': 'Sofina', 'link': 'https://www.mannings.com.hk/en/sofina-ip-skin-care-uv-protect-emulsion-02-oil-control-spf50-pa-30ml/p/285544', 'image': 'skincare_image/Screenshot 2025-07-13 at 12.22.22 PM.png', 'price': 220}
    ]
}

def get_hardcoded_recommendations(analysis_text: str) -> str:
    detected_conditions = []
    text_lower = analysis_text.lower()
    if 'dry' in text_lower:
        detected_conditions.append('dry skin')
    if 'oily' in text_lower:
        detected_conditions.append('oily skin')
    if 'acne' in text_lower:
        detected_conditions.append('acne')
    if 'dark circles' in text_lower or '黑眼圈' in text_lower:
        detected_conditions.append('dark circles')
    if 'wrinkles' in text_lower:
        detected_conditions.append('wrinkles')
    if 'sunburn' in text_lower or '曬傷' in text_lower:
        detected_conditions.append('sunburn')
    if 'healthy' in text_lower or 'no issues' in text_lower or 'no visible conditions' in text_lower:
        detected_conditions.extend(['face wash', 'sunscreen'])

    if not detected_conditions:
        return ''

    opening_phrases = [
        "\n\n## Product Recommendations from Mannings\nYou can consider buying these products from Mannings:\n",
        "\n\n## Mannings Product Suggestions\nHere are some great options available at Mannings that might help:\n",
        "\n\n## Your Recommended Products from Mannings\nBased on the analysis, you might want to check out these products at Mannings:\n",
        "\n\n## Skincare Picks from Mannings\nThinking about products? Here's a selection from Mannings that could be perfect for you:\n",
        "\n\n## Find These at Mannings\nYou can find these recommended products at Mannings to support your skincare routine:\n"
    ]
    recs = random.choice(opening_phrases)

    for condition in detected_conditions:
        if condition in products:
            recs += f'\n### For {condition.capitalize()}\n'
            for prod in products[condition]:
                # Start card - Product name is now just bold text
                card = f"**{prod.get('name', 'N/A')}**\n\n"
                
                # Convert image path to base64 URI and create clickable image
                image_uri = ""
                if prod.get('image'):
                    image_uri = image_to_base64_uri(prod['image'])

                if image_uri and prod.get('link'):
                    card += f"[![{prod.get('name', 'N/A')}]({image_uri})]({prod.get('link')})\n\n"
                elif image_uri: # Image without link if link is missing
                    card += f"![{prod.get('name', 'N/A')}]({image_uri})\n\n"
                
                # Details using <br> for line breaks
                details = []
                if prod.get("brand"):
                    details.append(f"**Brand:** {prod['brand']}")
                if prod.get("price"):
                    details.append(f"**Price:** HKD {prod['price']}")
                
                if details:
                    card += "<br>".join(details) + "\n\n"

                # Buy Link
                if prod.get('link'):
                     card += f"[Click here to buy]({prod.get('link')})\n"
                
                # Separator
                card += "\n---\n"
                recs += card
    return recs

SYSTEM_PROMPT_SKINCARE = """
You are a highly sophisticated AI skincare expert. Your goal is to provide a personalized skin analysis based on a user's image.

Your Process:
1.  **Analyze the provided image** to identify key skin conditions like dryness, oiliness, acne, dark circles, dark spots, or wrinkles. If the skin looks healthy, note that.
2.  **For EACH identified condition, provide**:
    a.  A `condition_name`.
    b.  A brief `analysis` of what you see.
    c.  The likely `causes`.
3.  **Provide a suggested skincare routine**:
    a.  Create a `morning_routine` with steps as an array of strings.
    b.  Create an `evening_routine` with steps as an array of strings.
4.  **Provide a general tip** for skin maintenance.

Output Format:
Respond ONLY with a single, well-formed JSON object. Do not include any text outside of the JSON.
{{
  "skin_conditions": [
    {{
      "condition_name": "e.g., Dry Skin",
      "analysis": "e.g., The skin on the cheeks appears flaky...",
      "causes": "e.g., Could be due to environmental factors or dehydration."
    }}
  ],
  "skincare_routine": {{
    "morning": ["Step 1: Cleanse with a gentle, hydrating cleanser.", "Step 2: Apply a vitamin C serum."],
    "evening": ["Step 1: Double cleanse to remove makeup and impurities.", "Step 2: Apply a retinol treatment."]
  }},
  "general_tip": "e.g., Remember to drink plenty of water throughout the day."
}}
"""

class SkincareAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="skincare_agent")

    async def _repair_json_with_llm(self, broken_json_string: str) -> str:
        """Uses a second LLM call to repair a broken JSON string."""
        repair_prompt = f"""The following text is supposed to be a single JSON object, but it is malformed. Please correct any syntax errors and return ONLY the valid JSON object.

Broken JSON:
```json
{broken_json_string}
```"""
        repair_llm = genai.GenerativeModel('gemini-1.5-flash-latest') # Cheaper model for fixing
        response = await repair_llm.generate_content_async(
            repair_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        llm = genai.GenerativeModel('gemini-1.5-pro-latest')
        repair_llm = genai.GenerativeModel('gemini-1.5-flash-latest') # Cheaper model for fixing
        image_base64 = ctx.session.state.get("image_base64")

        if not image_base64:
            content = genai_types.Content(role="model", parts=[genai_types.Part(text="I need an image to analyze skincare.")])
            yield Event(author=self.name, content=content)
            return

        pil_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        
        response = await llm.generate_content_async(
            [SYSTEM_PROMPT_SKINCARE, pil_image],
            generation_config={"response_mime_type": "application/json"}
        )
        
        response_text = response.text
        data = {}
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("SkincareAgent: Initial JSON parsing failed. Attempting to repair.")
            repaired_json_str = await self._repair_json_with_llm(response_text)
            try:
                data = json.loads(repaired_json_str)
            except json.JSONDecodeError as e:
                error_msg = f"Failed to decode JSON even after repair attempt: {e}"
                logger.error(error_msg)
                content = genai_types.Content(role="model", parts=[genai_types.Part(text=error_msg)])
                yield Event(author=self.name, content=content)
                return

        # --- Format JSON into Markdown Tables ---
        markdown_output = "##  Skin Analysis\n"
        conditions = data.get('skin_conditions', [])
        if conditions:
            markdown_output += "| Condition | Analysis | Potential Causes |\n"
            markdown_output += "|---|---|---|\n"
            for c in conditions:
                markdown_output += f"| **{c.get('condition_name', 'N/A')}** | {c.get('analysis', 'N/A')} | {c.get('causes', 'N/A')} |\n"
        else:
            markdown_output += "Your skin appears to be in great condition! No specific issues were detected.\n"

        markdown_output += "\n---\n"
        
        markdown_output += "## ‍⚕️ Suggested Skincare Routine\n"
        routine = data.get('skincare_routine', {})
        if routine:
             # Morning Routine
            markdown_output += "### Morning Routine\n"
            morning_steps = routine.get('morning', [])
            for i, step in enumerate(morning_steps, 1):
                markdown_output += f"{i}. {step}\n"
            
            # Evening Routine
            markdown_output += "\n### Evening Routine\n"
            evening_steps = routine.get('evening', [])
            for i, step in enumerate(evening_steps, 1):
                markdown_output += f"{i}. {step}\n"
        
        markdown_output += "\n---\n"
        
        tip = data.get('general_tip', '')
        if tip:
            markdown_output += f"** General Tip:** {tip}\n"

        # Combine analysis text for product recommendation keywords
        analysis_keywords = " ".join([c.get('condition_name', '') for c in conditions])
        if not analysis_keywords.strip():
            analysis_keywords = "healthy skin" # for sunscreen/face wash recs

        recs = get_hardcoded_recommendations(analysis_keywords)
        final_output = markdown_output + recs

        content = genai_types.Content(role="model", parts=[genai_types.Part(text=final_output)])
        yield Event(author=self.name, content=content)
