import json
import os

import dspy
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from .image_handler import pil_image_to_base64



class SkincareAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="skincare_agent",
            model=os.environ['GEMINI_MODEL_NAME'],
            instruction="Analyze facial images and provide skincare recommendations. Use tools if needed.",
            description="Agent for skincare analysis and recommendations.",
            tools=[google_search]
        )

    @observe()
    def analyze_face(self, image_base64: str) -> str:
        prompt_template = PromptTemplate(
            input_variables=["image"],
            template="""Analyze this facial image (base64: {image}) for conditions like dry/oily skin, dark eye circles, acne, dark spots.\n\nIf the skin appears healthy with no issues, start with positive compliments and focus on maintenance.\n\nProvide:\n- Detailed analysis of conditions (or confirmation of healthy skin).\n- Reasons/causes for each (or factors maintaining healthy skin).\n- Skincare routine suggestions (maintenance for healthy skin).\n- Product recommendations: Use google_search to find products online, include links and image URLs.\n\nFormat as Markdown text. Do not include disclaimers."""
        )
        chain = prompt_template | self.llm
        response = chain.invoke({"image": image_base64})
        text = response.content

        # Post-processing for healthy skin
        if 'healthy' in text.lower() or 'no issues' in text.lower() or 'no visible conditions' in text.lower():
            text = '## Positive Skin Assessment\nYour skin looks radiant, balanced, and healthy! Great job maintaining it.\n\n' + text
            text += '\n\n## Maintenance Tips\nTo keep your skin glowing: Use a gentle daily cleanser, moisturizer, and SPF. Consider antioxidants for prevention.\n'
            # Placeholder for hardcoded Mannings products (add here when provided)
            text += '\n\n## Recommended Maintenance Products\n(Coming soon: Hardcoded suggestions from Mannings with links and images)\n'

        # Extract conditions from response to search products
        # Simple extraction (improve as needed)
        conditions = text.split('##')[1] if '##' in text else 'skin conditions'
        search_query = f"best skincare products for {conditions}"
        search_results = google_search(search_query)

        output = text + "\n\n## Product Recommendations\n"
        for item in search_results.get('items', [])[:3]:  # Limit to 3
            output += f"- {item.get('title', 'Product')} ({item.get('link', '')})\n  ![Image]({item.get('pagemap', {}).get('cse_image', [{}])[0].get('src', '')})\n"
        return output
