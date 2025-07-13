import asyncio
import base64
import json
import logging
import os
import re
from typing import Tuple

import gradio as gr
import google.generativeai as genai
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to encode local image to base64 data URI
def image_to_base64_uri(file_path):
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


# --- Product Definitions ---
# This data is now global to be accessible for the prompt and rendering.
ALL_PRODUCTS_FOR_PROMPT = [
    {
        "product_name": "Sebamed Baby Shampoo 500ml",
        "description": "Extra mild cleansing for babies and children hair and scalp."
    },
    {
        "product_name": "Head & Shoulders Smooth & Silky Anti-Dandruff Shampoo",
        "description": "Goes deep into the scalp to remove dandruff and protect the scalp."
    },
    {
        "product_name": "Moisturizing & Styling Hair Gel",
        "description": "Moisturizes and provides hold for voluminous and lustrous hair."
    },
    {
        "product_name": "Nourishing Hair Lotion",
        "description": "Moisturizing but not greasy, nourishes silky honey hair."
    },
    {
        "product_name": "Hair Styling Finishing Spray",
        "description": "Holds your hair style with a strong, finishing lock."
    },
    {
        "product_name": "Hair Repair Conditioner",
        "description": "Replenishes moisture and shine to damaged hair."
    },
    {
        "product_name": "Purifying Hair & Scalp Serum",
        "description": "Helps with scalp blemishes and purifies pores."
    }
]


# Load environment variables from .env file


# Configure google-generativeai for the ADK Agent and other calls
try:
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        genai.configure(api_key=google_api_key)  # type: ignore
        logger.info("google-generativeai configured successfully.")
    else:
        logger.warning("GOOGLE_API_KEY not set. The application will not work.")
except Exception as e:
    logger.error(f"Failed to configure google-generativeai: {e}")


# --- System Prompt ---
SYSTEM_PROMPT_HAIRCUT = """
You are a highly sophisticated AI stylist. Your goal is to provide personalized haircut recommendations based on an image analysis and user preferences.

Your Process:
1.  **Analyze the provided "Image Analysis" text** which describes the user's face shape, current hair, etc.
2.  **Consider User Preferences** to understand their goals.
3.  **Determine the number of suggestions (N)**: Check the "Specific Requirements" for a requested number (e.g., 'give me 5 suggestions'). If an integer is specified, use that as N (limit to 1-10). Otherwise, default to 2-3 suggestions.
4.  **Generate Recommendations**: Based on all available information, suggest N specific, flattering, and fashionable haircut styles.
5.  **For EACH suggestion, you must**:
    a.  Provide the `style_name`.
    b.  Write a brief, encouraging `description`.
    c.  In the `suitable_venue` field, describe the social settings or occasions where this hairstyle is most appropriate (e.g., 'professional office setting,' 'casual weekend outings,' 'edgy concert vibe').
    d.  Provide a `method` for styling.
    e.  Offer `communication_feedbacks` for a hairstylist.
    f.  Estimate the `duration` the style will last or take to grow.
    g.  From the 'Available Products' list below, recommend one or two products by name that are most suitable for creating this style. Put them in a `recommended_products` array.
6.  **General Tip**: Provide one relevant general hair care `general_tip`.

{available_products_text}

Output Format:
Respond ONLY with a single, well-formed JSON object. Do not include any text outside of the JSON.
{{
  "style_suggestions": [
    {{
      "style_name": "...",
      "suitable_venue": "...",
      "description": "...",
      "method": "...",
      "communication_feedbacks": "...",
      "duration": "...",
      "recommended_products": ["Product Name 1", "Product Name 2"]
    }}
  ],
  "general_tip": "..."
}}
"""


# --- ADK Agent Setup ---
session_service = InMemorySessionService()

# The agent is defined globally, but the prompt will be formatted inside the request function
haircut_agent = LlmAgent(
    name="haircut_advisor",
    model="gemini-1.5-pro-latest",
    # The instruction will be set dynamically in `suggest_haircut_async`
)

runner = Runner(
    agent=haircut_agent,
    app_name="haircut_app",
    session_service=session_service
)


# --- Core Functions ---
async def analyze_image(pil_image: Image.Image) -> str:
    """Analyzes the user's image to describe their features."""
    if pil_image is None:
        return "No image provided."

    model = genai.GenerativeModel('gemini-1.5-pro-latest')  # type: ignore

    analysis_prompt = """
    Analyze the uploaded image to accurately determine the user’s face shape (e.g., oval, square, round, heart, long).
    Describe the user’s CURRENT haircut: identify its style, length, and apparent hair texture.
    Comment on potential hair health problems visible in the image.
    Respond in a concise paragraph.
    """
    
    response = await model.generate_content_async(
        [analysis_prompt, pil_image],
        generation_config={
            "candidate_count": 1,
            "max_output_tokens": 1024, # Increased token limit
            "temperature": 0.5,
        }
    )
    
    logger.info(f"Image analysis response finish reason: {response.candidates[0].finish_reason}")
    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text
    return "Could not get a valid analysis from the model."


async def repair_json_with_llm(broken_json_string: str) -> str:
    """
    Uses a second LLM call to repair a broken JSON string.
    """
    logger.info("Attempting to repair malformed JSON with a second LLM call...")
    logger.debug(f"--- Broken JSON --- \n{broken_json_string}\n--------------------")
    repair_prompt = f"""The following text is supposed to be a single JSON object, but it is malformed. Please correct any syntax errors (e.g., missing commas, extra characters, unclosed brackets) and return ONLY the valid JSON object. Do not add any text before or after the JSON.

Broken JSON:
```json
{broken_json_string}
```"""

    try:
        model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')  # type: ignore
        response = await model.generate_content_async(
            repair_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        repaired_json = response.text
        logger.info("Successfully repaired JSON.")
        logger.debug(f"--- Repaired JSON --- \n{repaired_json}\n--------------------")
        return repaired_json
    except Exception as e:
        logger.error(f"Failed to repair JSON: {e}")
        return broken_json_string


async def suggest_haircut_async(pil_image: Image.Image, exp_length: str, exp_style: str, exp_venue: str, sex: str, region_religion: str, exp_overall: str, other_problems: str, specific_req: str) -> Tuple[str, str]:
    logger.info("--- Starting new haircut suggestion request ---")
    # Step 1: Analyze the user's image to get a text description.
    logger.info("Step 1: Analyzing user image...")
    image_analysis = await analyze_image(pil_image)
    if "error" in image_analysis.lower() or "Could not get" in image_analysis:
        logger.error(f"Image analysis failed: {image_analysis}")
        return image_analysis, ""
    logger.info(f"Image analysis successful:\n{image_analysis}")

    # --- Create Dynamic Prompt for AI ---
    # Create the text block of available products for the prompt
    available_products_list = [
        f"- {prod['product_name']}: {prod['description']}" for prod in ALL_PRODUCTS_FOR_PROMPT
    ]
    available_products_text = "Available Products:\n" + "\n".join(available_products_list)
    
    # Format the final system prompt
    final_system_prompt = SYSTEM_PROMPT_HAIRCUT.format(available_products_text=available_products_text)
    
    # Update agent's instruction for this call
    haircut_agent.instruction = final_system_prompt

    # Step 2: Use the ADK agent to get text-based suggestions.
    logger.info("Step 2: Calling stylist agent for suggestions...")
    user_preferences = f"Image Analysis:\n{image_analysis}\n\nPersonal Preferences:\nExpected Length: {exp_length}\nExpected Style: {exp_style}\nExpected Venue: {exp_venue}\nSex: {sex}\nRegion/Religion: {region_religion} (consider cultural/historical aspects)\nExpected Overall Style: {exp_overall}\nOther Problems: {other_problems}\nSpecific Requirements: {specific_req}"
    full_prompt = user_preferences + "\nPlease suggest new haircuts based on the above analysis and preferences."

    user_content = types.Content(role="user", parts=[types.Part(text=full_prompt)])
    user_id = "user1"
    session_id = "session1"
    try:
        await session_service.create_session(app_name="haircut_app", user_id=user_id, session_id=session_id, state={})
    except ValueError:
        logger.info("Session already exists, reusing.")
        pass # Session already exists, which is fine for this app

    events = runner.run_async(user_id=user_id, session_id=session_id, new_message=user_content)
    response_text = ""
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            response_text += (event.content.parts[0].text or '')

    logger.info(f"--- Raw Stylist Agent Response --- \n{response_text}\n--------------------------")

    # Step 3: Parse the response and find image URLs.
    logger.info("Step 3: Parsing agent response...")
    data = {}
    json_string = ""
    try:
        # Extract JSON from the response text, which might be wrapped in markdown
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
            data = json.loads(json_string)
        else:
            logger.warning("No JSON object found in the raw response.")
            # If no JSON is found, we'll let it fall through to the repair mechanism
            raise json.JSONDecodeError("No JSON object found in response.", response_text, 0)

    except json.JSONDecodeError:
        logger.warning("Initial JSON parsing failed. Attempting to repair.")
        try:
            # Use the full response text for repair, as it might have context
            repaired_json_str = await repair_json_with_llm(response_text)
            data = json.loads(repaired_json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON even after repair attempt: {e}\nRaw response: {response_text}")
            return "Error: Could not parse the AI's response. The format was invalid.", ""

    try:
        # --- Format Output into two separate Markdown strings ---

        # Create data URI for local product images
        sebamed_image_path = "/Users/votee/Desktop/Screenshot 2025-07-12 at 6.10.05 PM.png"
        gel_image_path = "/Users/votee/Desktop/Screenshot 2025-07-12 at 6.19.00 PM.png"
        lotion_image_path = "/Users/votee/Desktop/Screenshot 2025-07-12 at 6.35.58 PM.png"
        spray_image_path = "/Users/votee/Desktop/Screenshot 2025-07-12 at 6.39.47 PM.png"
        conditioner_image_path = "/Users/votee/Desktop/Screenshot 2025-07-12 at 6.41.53 PM.png"
        serum_image_path = "/Users/votee/Desktop/Screenshot 2025-07-12 at 6.44.30 PM.png"
        sebamed_image_uri = image_to_base64_uri(sebamed_image_path)
        gel_image_uri = image_to_base64_uri(gel_image_path)
        lotion_image_uri = image_to_base64_uri(lotion_image_path)
        spray_image_uri = image_to_base64_uri(spray_image_path)
        conditioner_image_uri = image_to_base64_uri(conditioner_image_path)
        serum_image_uri = image_to_base64_uri(serum_image_path)


        # --- Define Products ---
        # Base products that are always shown
        base_products = [
            {
                "product_name": "Sebamed Baby Shampoo 500ml",
                "product_link": "https://www.mannings.com.hk/en/sebamed-baby-shampoo-500ml/p/598995",
                "image_url": sebamed_image_uri,
                "description": (
                    "Extra mild cleansing for babies and children hair and scalp. "
                    "Doesn’t sting in eyes. Promotes protective acid mantle of the scalp."
                )
            },
            {
                "product_name": "Head & Shoulders Smooth & Silky Anti-Dandruff Shampoo",
                "product_link": "https://www.mannings.com.hk/en/head-n-shoulders-smooth-n-silky-shampoo-750ml/p/155700",
                "image_url": "https://images.ctfassets.net/cfexf643femw/4n43yF3138f5f6o43p0YpE/8f0376c34b3ed188e0b29845ab264d2b/750-anti-dandruff-shampoo-smooth-silky.png",
                "description": "Goes deep into the scalp to remove dandruff and protect the scalp, leaving hair feeling soft and smooth."
            }
        ]

        # Conditional products
        hair_gel_product = {
            "product_name": "Moisturizing & Styling Hair Gel",
            "product_link": "https://www.mannings.com.hk/en/gatsby-styling-gel-super-hard-200g/p/106606", # Placeholder link
            "image_url": gel_image_uri,
            "description": (
                "Moisturizes to the ends of the hair for voluminous and lustrous hair. "
                "Contains 3 types of penetrating collagen with different molecular weights. "
                "'Deep Moisturizing Night Care Formula'. Three major formulas without additives. For damaged scalp."
            )
        }
        hair_lotion_product = {
            "product_name": "Nourishing Hair Lotion",
            "product_link": "https://www.mannings.com.hk/en/lucido-l-hair-lotion-moist-200ml/p/122472", # Placeholder link
            "image_url": lotion_image_uri,
            "description": (
                "The texture is silky and light. It is absorbed instantly upon application. "
                "It is moisturizing but not greasy, and nourishes silky honey hair."
            )
        }
        hair_spray_product = {
            "product_name": "Hair Styling Finishing Spray",
            "product_link": "https://www.mannings.com.hk/en/lucido-l-d-keep-hair-spray-sh-180g/p/122474", # Placeholder link
            "image_url": spray_image_uri,
            "description": "Holds your hair style."
        }
        hair_conditioner_product = {
            "product_name": "Hair Repair Conditioner",
            "product_link": "https://www.mannings.com.hk/en/ichikami-airy-n-silky-conditioner-pump-480g/p/597871", # Placeholder link
            "image_url": conditioner_image_uri,
            "description": (
                "Pure Japanese Botanical Essences. Replenishes moisture and shine to damaged hair. "
                "Seals damaged cuticles that are prone to peeling due to friction, while preventing future damage. "
                "Leaving hair smooth and airy."
            )
        }
        hair_serum_product = {
            "product_name": "Purifying Hair & Scalp Serum",
            "product_link": "https://www.mannings.com.hk/en/loreal-paris-extra-oil-serum-100ml/p/162879", # Placeholder link
            "image_url": serum_image_uri,
            "description": (
                "Suitable for normal/oily scalps. First choice for effective scalp purification and brightening. "
                "Helps with scalp blemishes and tightens pores. A home treatment to first deeply purify the scalp, "
                "then deeply moisturize and repair hair. No alcohol, additives, and paraben."
            )
        }
        
        # --- Build Product List with Personalized Explanations ---
        all_products = list(base_products)
        search_text = (other_problems + " " + exp_style + " " + image_analysis).lower()

        # Configuration for conditional products
        conditional_products_config = [
            {
                "product": hair_gel_product,
                "keywords": ['gel', 'hold', 'styling', 'frizz', 'flyaways', 'style'],
                "reason": "provide hold and control for styling."
            },
            {
                "product": hair_lotion_product,
                "keywords": ['lotion', 'moisture', 'dry', 'nourish', 'brittle'],
                "reason": "address dryness and provide moisture."
            },
            {
                "product": hair_spray_product,
                "keywords": ['spray', 'finish', 'set', 'lock'],
                "reason": "lock in your hairstyle."
            },
            {
                "product": hair_conditioner_product,
                "keywords": ['conditioner', 'damage', 'repair', 'smooth', 'shine'],
                "reason": "repair damage and improve smoothness."
            },
            {
                "product": hair_serum_product,
                "keywords": ['serum', 'scalp', 'impurities', 'blemishes', 'purify', 'pores'],
                "reason": "purify the scalp and address blemishes."
            }
        ]
        
        for config in conditional_products_config:
            # Find the first keyword that was matched to provide a specific reason
            matched_keyword = next((keyword for keyword in config['keywords'] if keyword in search_text), None)
            if matched_keyword:
                product_to_add = config['product'].copy()
                product_to_add['personalized_reason'] = f"Because your needs seem to include '{matched_keyword}', this product is recommended to help {config['reason']}"
                all_products.append(product_to_add)


        # 1. Analysis and Products Markdown
        product_markdown = "### Recommended Products\n"
        for prod in all_products:
            product_markdown += f"**{prod.get('product_name', 'N/A')}**\n"
            if prod.get('image_url') and prod.get('product_link'):
                product_markdown += f"[![{prod.get('product_name', 'N/A')}]({prod.get('image_url')})]({prod.get('product_link')})\n"
                if prod.get('description'):
                    product_markdown += f"\n{prod.get('description')}\n"
                if prod.get('personalized_reason'):
                    product_markdown += f"\n**Why it's recommended for you:** {prod.get('personalized_reason')}\n"
                product_markdown += f"\n[Click here to buy]({prod.get('product_link')})\n\n---\n"
            else:
                logger.warning(f"Skipping incomplete product data: {prod}")
        
        analysis_and_products_markdown = f"## Current Hair Analysis\n{image_analysis}\n\n---\n\n{product_markdown}"

        # 2. Interleaved Style Suggestions Markdown
        suggestions = data.get('style_suggestions', [])
        suggestions_markdown = "## Style Recommendations\n"
        if not suggestions:
            suggestions_markdown += "No style suggestions were generated."
        else:
            for i, sug in enumerate(suggestions, 1):
                suggestions_markdown += f"### Recommendation #{i}: {sug.get('style_name', 'N/A')}\n\n"
                
                # Create a markdown table for the details
                suggestions_markdown += "| Feature | Details |\n"
                suggestions_markdown += "|---|---|\n"
                suggestions_markdown += f"| **Suitable Venue** | {sug.get('suitable_venue', 'N/A')} |\n"
                suggestions_markdown += f"| **Description** | {sug.get('description', 'N/A')} |\n"
                suggestions_markdown += f"| **Method** | {sug.get('method', 'N/A')} |\n"
                suggestions_markdown += f"| **Communication for Stylist** | {sug.get('communication_feedbacks', 'N/A')} |\n"
                
                # Add recommended products for the style, if any
                recommended_products = sug.get('recommended_products')
                if recommended_products:
                    suggestions_markdown += f"| **Recommended Products** | {', '.join(recommended_products)} |\n"

                suggestions_markdown += f"| **Duration** | {sug.get('duration', 'N/A')} |\n\n---\n"
        
        suggestions_markdown += f"\n**General Tip:** {data.get('general_tip', 'N/A')}"

        logger.info("--- Haircut suggestion request finished successfully ---")
        return analysis_and_products_markdown, suggestions_markdown

    except Exception as e:
        logger.error(f"Error processing suggestions: {e}")
        return f"An error occurred: {str(e)}", ""


def suggest_haircut(pil_image: Image.Image, exp_length: str, exp_style: str, exp_venue: str, sex: str, region_religion: str, exp_overall: str, other_problems: str, specific_req: str) -> Tuple[str, str]:
    return asyncio.run(suggest_haircut_async(pil_image, exp_length, exp_style, exp_venue, sex, region_religion, exp_overall, other_problems, specific_req))


if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.Markdown("# AI Haircut Advisor")
        gr.Markdown("Upload a photo of yourself to get personalized haircut recommendations from our AI stylist!")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Your Image")
                exp_length_input = gr.Textbox(label="Expected Length (Short or Long)")
                exp_style_input = gr.Textbox(label="Expected Style")
                exp_venue_input = gr.Textbox(label="Expected Venue")
                sex_input = gr.Textbox(label="Sex")
                region_religion_input = gr.Textbox(label="Region/Religion")
                exp_overall_input = gr.Textbox(label="Expected Overall Style")
                other_problems_input = gr.Textbox(label="Other Problems or Needs (e.g., 'need hair gel')")
                specific_req_input = gr.Textbox(label="Specific Requirements (e.g., 'give me 10 suggestions')")
                submit_button = gr.Button("Get Recommendations")
            with gr.Column():
                with gr.Group():
                    analysis_output = gr.Markdown(
                        label="Hair Analysis & Recommended Products",
                        value="*Your analysis and product recommendations will appear here...*"
                    )
                with gr.Group():
                    suggestions_output = gr.Markdown(
                        label="Style Recommendations",
                        value="*Your style recommendations will appear here...*"
                    )

        submit_button.click(
            fn=suggest_haircut,
            inputs=[image_input, exp_length_input, exp_style_input, exp_venue_input, sex_input, region_religion_input, exp_overall_input, other_problems_input, specific_req_input],
            outputs=[analysis_output, suggestions_output]
        )

    demo.launch() 