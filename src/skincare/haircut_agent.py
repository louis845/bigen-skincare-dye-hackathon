import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import re
from typing import AsyncGenerator, List, Tuple

from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types as genai_types
import google.generativeai as genai
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Configure google-generativeai
try:
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        genai.configure(api_key=google_api_key)
    else:
        logging.warning("GOOGLE_API_KEY not set. The application will not work.")
except Exception as e:
    logging.error(f"Failed to configure google-generativeai: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def image_to_base64_uri(filepath: str) -> str:
    """Converts an image file to a base64 data URI."""
    try:
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            mime_type = mimetypes.guess_type(filepath)[0] or 'image/png'
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        logger.error(f"Image file not found: {filepath}")
        return ""
    except Exception as e:
        logger.error(f"Error encoding image {filepath}: {e}")
        return ""

# This function is no longer needed as we are moving to a simpler markdown format.
# def format_product_html(product: dict) -> str:
#     """Formats a single product dictionary into an HTML string."""
#     return f"""
#     <div class="product-card">
#         <a href="{product['product_link']}" target="_blank" class="product-image-link">
#             <img src="{product['image_url']}" alt="{product.get('product_name_en', 'Product Image')}">
#         </a>
#         <div class="product-info">
#             <div class="product-name-zh">{product.get('product_name_zh', '')}</div>
#             <div class="product-name-en">{product.get('product_name_en', '')}</div>
#             <div class="product-brand">by: {product.get('brand', '')}</div>
#             <div class="product-price">HK$ {product.get('price', 'N/A')}</div>
#             <a href="{product['product_link']}" class="product-button" target="_blank">View Product</a>
#         </div>
#     </div>
#     """

# --- Product Definitions ---
# Single source of truth for all products.
ALL_PRODUCTS = [
    {
        "product_name_zh": "Head & Shoulders海倫仙度絲去屑絲滑柔順洗髮乳 750克 + 控油蓬鬆洗髮乳 350克",
        "product_name_en": "Head & Shoulders Silky Soft Anti-Dandruff Shampoo 750g + Kingsman Oil Control Shampoo 350g",
        "brand": "Head & Shoulders 海倫仙度絲",
        "product_link": "https://www.mannings.com.hk/zh-hant/head-shoulders-silky-soft-anti-dandruff-shampoo-750g-kingsman-oil-control-shampoo-350g/p/395970",
        "image_path": "haircut_image/Screenshot 2025-07-12 at 6.09.09 PM.png",
        "price": "89.90",
        "description": "Goes deep into the scalp to remove dandruff and protect the scalp, leaving hair feeling soft and smooth.",
        "keywords": ["shampoo", "dandruff", "oily", "smooth"]
    },
    {
        "product_name_zh": "Sebamed 施巴嬰幼童洗髮露 500毫升",
        "product_name_en": "Sebamed Baby Shampoo 500ml",
        "brand": "Sebamed 施巴",
        "product_link": "https://www.mannings.com.hk/zh-hant/sebamed-baby-shampoo-500ml/p/598995",
        "image_path": "haircut_image/Screenshot 2025-07-12 at 6.10.05 PM.png",
        "price": "154",
        "description": "Extra mild cleansing for babies and children hair and scalp. Doesn’t sting in eyes.",
        "keywords": ["shampoo", "mild", "sensitive", "baby"]
    },
    {
        "product_name_zh": "Yolu 夜間柔順修復凝膠髮膜香梨和天竺葵香味145克",
        "product_name_en": "Yolu Relax Night Repair Gel Hair Mask 145g",
        "brand": "Yolu",
        "product_link": "https://www.mannings.com.hk/zh-hant/yolu-relax-night-repair-gel-hair-mask-145g/p/397380",
        "image_path": "haircut_image/Screenshot 2025-07-12 at 6.19.00 PM.png",
        "price": "129",
        "description": "A gel hair mask for night-time repair, leaving hair smooth.",
        "keywords": ["lotion", "mask", "repair", "night", "smooth"]
    },
    {
        "product_name_zh": "Honeyque蜂蜜蛋白晚間修護潤髮乳液 150毫升",
        "product_name_en": "Honeyque Night Repair Hair Milk 150ml",
        "brand": "Honeyque",
        "product_link": "https://www.mannings.com.hk/zh-hant/honeyque-night-repair-hair-milk-150ml/p/057794",
        "image_path": "haircut_image/Screenshot 2025-07-12 at 6.35.58 PM.png",
        "price": "158",
        "description": "A hair milk that provides night-time repair with honey proteins.",
        "keywords": ["lotion", "milk", "repair", "night", "honey"]
    },
    {
        "product_name_zh": "Liese Sifone彈性定型噴霧 160毫升",
        "product_name_en": "Liese Sifone Hair Spray 160ml",
        "brand": "Liese Sifone",
        "product_link": "https://www.mannings.com.hk/zh-hant/liese-sifone-hair-spray-160ml/p/740811",
        "image_path": "haircut_image/Screenshot 2025-07-12 at 6.39.47 PM.png",
        "price": "39.9",
        "description": "A finishing spray to hold your hairstyle with flexibility.",
        "keywords": ["spray", "hold", "styling", "finish"]
    },
    {
        "product_name_zh": "50 Megumi 50惠養潤豐盈護髮素滋養型 400毫升",
        "product_name_en": "50 Megumi Moist Conditioner 400ml",
        "brand": "50 Megumi 50 惠",
        "product_link": "https://www.mannings.com.hk/zh-hant/50-megumi-moist-conditioner-400ml/p/569020",
        "image_path": "haircut_image/Screenshot 2025-07-13 at 3.47.02 PM.png",
        "price": "115.9",
        "description": "A nourishing conditioner designed to improve hair volume and health.",
        "keywords": ["conditioner", "moist", "volume", "nourishing"]
    },
    {
        "product_name_zh": "Ichikami 柔韌順滑護髮素 480克",
        "product_name_en": "Ichikami Smoothing Conditioner 480g",
        "brand": "Ichikami",
        "product_link": "https://www.mannings.com.hk/en/ichikami-smoothing-conditioner-480g/p/067728",
        "image_path": "haircut_image/Screenshot 2025-07-12 at 6.41.53 PM.png",
        "price": "89",
        "description": "A smoothing conditioner with Japanese botanical essences for silky hair.",
        "keywords": ["conditioner", "smoothing", "silky", "botanical"]
    },
    {
        "product_name_zh": "Gatsby 定型啫喱膏 特硬 200克",
        "product_name_en": "Gatsby Super Hard Styling Gel 200g",
        "brand": "Gatsby",
        "product_link": "https://www.mannings.com.hk/zh-hant/gatsby-super-hard-styling-gel-200g/p/233064",
        "image_path": "haircut_image/Screenshot 2025-07-13 at 3.55.34 PM.png",
        "price": "39.9",
        "description": "A super hard styling gel for strong hold.",
        "keywords": ["gel", "styling", "hold", "hard"]
    },
    {
        "product_name_zh": "Gatsby強效髮泥 30克",
        "product_name_en": "Gatsby Technical Design Clay 30g",
        "brand": "Gatsby",
        "product_link": "https://www.mannings.com.hk/zh-hant/gatsby-technical-design-clay-30g/p/963983",
        "image_path": "haircut_image/Screenshot 2025-07-13 at 3.55.02 PM.png",
        "price": "43",
        "description": "A design clay for creating textured and defined styles.",
        "keywords": ["gel", "clay", "styling", "texture", "matte"]
    },
    {
        "product_name": "Purifying Hair & Scalp Serum",
        "product_link": "https://www.mannings.com.hk/en/loreal-paris-extra-oil-serum-100ml/p/162879",
        "image_path": "haircut_image/Screenshot 2025-07-12 at 6.44.30 PM.png",
        "description": "Helps with scalp blemishes and purifies pores.",
        "keywords": ["serum", "scalp", "purify", "pores"],
        "price": "N/A",
        "brand": "L'Oreal Paris",
        "product_name_en": "Purifying Hair & Scalp Serum",
        "product_name_zh": "淨化頭皮精華"
    }
]

# Process images and create a lookup dictionary
# This should be done once when the module is loaded.
PRODUCT_LOOKUP = {
    p["product_name_en"]: {**p, "image_url": image_to_base64_uri(p["image_path"])}
    for p in ALL_PRODUCTS
}


# --- System Prompt ---
SYSTEM_PROMPT_HAIRCUT = """
You are a highly sophisticated AI stylist. Your goal is to provide personalized haircut recommendations based on an image analysis and user preferences.

Your Process:
1.  **Analyze the provided "Image Analysis" text** which describes the user's face shape, current hair, etc.
2.  **Consider User Preferences** to understand their goals.
3.  **Determine the number of suggestions (N)**: Check the "Specific Requirements" for a requested number (e.g., 'give me 5 suggestions'). If an integer is specified, use that as N (limit to 1-5). Otherwise, default to 2-3 suggestions.
4.  **Generate Recommendations**: Based on all available information, suggest N specific, flattering, and fashionable haircut styles.
5.  **For EACH suggestion, you must**:
    a.  Provide the `style_name`.
    b.  Write a brief, encouraging `description`.
    c.  In the `suitable_venue` field, describe the social settings or occasions where this hairstyle is most appropriate (e.g., 'professional office setting,' 'casual weekend outings,' 'edgy concert vibe').
    d.  Provide a `method` for styling.
    e.  Offer `communication_feedbacks` for a hairstylist.
    f.  Estimate the `duration` the style will last or take to grow.
    g.  From the 'Available Products' list below, recommend one or two products by their exact English name (`product_name_en`) that are most suitable for creating or maintaining this style. Put them in a `recommended_products` array.
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
      "recommended_products": ["Product Name EN 1", "Product Name EN 2"]
    }}
  ],
  "general_tip": "..."
}}
"""


class HaircutAgent(BaseAgent):
    """An agent that provides haircut recommendations."""

    def __init__(self):
        super().__init__(name="haircut_agent")

    async def _analyze_image(self, image_base64: str) -> str:
        """Analyzes the user's image to describe their features."""
        if not image_base64:
            return "No image provided."
        
        analysis_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        pil_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))

        analysis_prompt = """
        Analyze the uploaded image to accurately determine the user’s face shape (e.g., oval, square, round, heart, long).
        Describe the user’s CURRENT haircut: identify its style, length, and apparent hair texture.
        Comment on potential hair health problems visible in the image.
        Respond in a concise paragraph in Traditional Chinese.
        """
        response = await analysis_model.generate_content_async([analysis_prompt, pil_image])
        
        try:
            return response.text
        except (ValueError, IndexError):
             return "無法分析圖片。請再試一次。(Could not get a valid analysis from the model.)"


    async def _repair_json_with_llm(self, broken_json_string: str) -> str:
        """Uses a second LLM call to repair a broken JSON string."""
        logger.info("Attempting to repair malformed JSON...")
        repair_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        repair_prompt = f"""The following text is supposed to be a single JSON object, but it is malformed. Please correct any syntax errors and return ONLY the valid JSON object.

Broken JSON:
```json
{broken_json_string}
```"""
        try:
            response = await repair_model.generate_content_async(
                repair_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return response.text
        except Exception as e:
            logger.error(f"JSON repair failed: {e}")
            return "{}"

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info("--- Starting new haircut suggestion request ---")
        image_base64 = ctx.session.state.get("image_base64")
        user_message = ctx.session.state.get("user_message", "Suggest a haircut for me.")

        if not image_base64:
            content = genai_types.Content(role="model", parts=[genai_types.Part(text="我需要一張圖片才能提供髮型建議。(I need an image to suggest a haircut.)")])
            yield Event(author=self.name, content=content)
            return

        logger.info("Step 1: Analyzing user image...")
        image_analysis = await self._analyze_image(image_base64)
        logger.info(f"Image analysis successful:\n{image_analysis}")
        
        # Create the text block of available products for the prompt
        available_products_list = [
            f"- {p['product_name_en']}: {p['description']}" for p in PRODUCT_LOOKUP.values()
        ]
        available_products_text = "Available Products:\n" + "\n".join(available_products_list)
        final_system_prompt = SYSTEM_PROMPT_HAIRCUT.format(available_products_text=available_products_text)

        full_prompt = f"Image Analysis:\n{image_analysis}\n\nUser Preferences:\n{user_message}"
        
        logger.info("Step 2: Calling stylist agent for suggestions...")
        suggestion_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        try:
            response = await suggestion_model.generate_content_async(
                [final_system_prompt, full_prompt],
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = response.text
        except Exception as e:
            logger.error(f"Stylist agent call failed: {e}")
            content = genai_types.Content(role="model", parts=[genai_types.Part(text="抱歉，AI造型師暫時無法回應。(Sorry, the AI stylist is currently unavailable.)")])
            yield Event(author=self.name, content=content)
            return
            
        logger.info("Step 3: Parsing agent response...")
        data = {}
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("Initial JSON parsing failed. Attempting to repair.")
            repaired_json_str = await self._repair_json_with_llm(response_text)
            try:
                data = json.loads(repaired_json_str)
            except json.JSONDecodeError as e:
                error_msg = f"Failed to decode JSON even after repair attempt: {e}"
                logger.error(error_msg)
                final_html = f"<p>抱歉，解析AI回應時發生錯誤。(Sorry, there was an error parsing the AI response.)</p><p>Debug Info: {error_msg}</p>"
                content = genai_types.Content(role="model", parts=[genai_types.Part(text=final_html)])
                yield Event(author=self.name, content=content)
                return

        # --- Build final Markdown output ---

        # Analysis Section
        final_markdown = f"## 髮質分析 (Hair Analysis)\n{image_analysis}\n\n---\n\n"

        # Style Recommendations Section
        suggestions = data.get('style_suggestions', [])
        all_recommended_products = set()
        
        final_markdown += "## 造型建議 (Style Recommendations)\n"
        if not suggestions:
            final_markdown += "沒有生成造型建議。(No style suggestions were generated.)\n"
        else:
            for i, sug in enumerate(suggestions, 1):
                final_markdown += f"### 建議 #{i}: {sug.get('style_name', 'N/A')}\n\n"
                final_markdown += "| Feature | Details |\n"
                final_markdown += "|---|---|\n"
                final_markdown += f"| **適合場合 (Venue)** | {sug.get('suitable_venue', 'N/A')} |\n"
                final_markdown += f"| **描述 (Description)** | {sug.get('description', 'N/A')} |\n"
                final_markdown += f"| **造型方法 (Method)** | {sug.get('method', 'N/A')} |\n"
                final_markdown += f"| **給造型師的建議 (Tips for Stylist)** | {sug.get('communication_feedbacks', 'N/A')} |\n"
                final_markdown += f"| **持續時間 (Duration)** | {sug.get('duration', 'N/A')} |\n\n"
                
                # Collect all unique recommended product names
                for prod_name in sug.get('recommended_products', []):
                    all_recommended_products.add(prod_name)
        final_markdown += "\n---\n"
        
        # Recommended Products Section
        final_markdown += "## 推薦產品 (Recommended Products)\n"
        if not all_recommended_products:
            final_markdown += "沒有特別推薦的產品。(No specific products were recommended.)\n"
        else:
            product_cards_html = ""
            for prod_name in sorted(list(all_recommended_products)):
                product_data = PRODUCT_LOOKUP.get(prod_name)
                if product_data:
                    card = f"**{product_data.get('product_name_zh', '')} ({product_data.get('product_name_en', 'N/A')})**\n\n"
                    image_uri = product_data.get('image_url', '')
                    
                    if image_uri and product_data.get('product_link'):
                        card += f"[![{product_data.get('product_name_en', 'N/A')}]({image_uri})]({product_data.get('product_link')})\n\n"
                    elif image_uri:
                        card += f"![{product_data.get('product_name_en', 'N/A')}]({image_uri})\n\n"
                    
                    details = []
                    if product_data.get("brand"):
                        details.append(f"**Brand:** {product_data['brand']}")
                    if product_data.get("price"):
                        details.append(f"**Price:** HK$ {product_data['price']}")
                    
                    if details:
                        card += "<br>".join(details) + "\n\n"

                    if product_data.get('product_link'):
                         card += f"[Click here to buy]({product_data.get('product_link')})\n"
                    
                    card += "\n---\n"
                    product_cards_html += card
                else:
                    logger.warning(f"Product '{prod_name}' recommended by LLM but not found in lookup.")
            
            if product_cards_html:
                 final_markdown += product_cards_html
            else:
                 final_markdown += "沒有找到對應的推薦產品。(Could not find the recommended products.)\n"
        
        final_markdown += "\n---\n"

        # General Tip Section
        if data.get('general_tip'):
            final_markdown += f"**護理小貼士 (General Tip):** {data.get('general_tip')}\n"

        logger.info("--- Haircut suggestion request finished successfully ---")
        content = genai_types.Content(role="model", parts=[genai_types.Part(text=final_markdown)])
        yield Event(author=self.name, content=content) 