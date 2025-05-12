import requests
import json
import asyncio
import edge_tts
import os
import tempfile
from playsound import playsound
from huggingface_hub import InferenceClient
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---
LLM_URL = os.getenv("LLM_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
hf_client = InferenceClient(api_key=HF_TOKEN)

# --- Helper Functions ---

def is_image_prompt(prompt):
    keywords = ["draw", "image", "visualize", "show me", "generate image", "picture", "illustrate"]
    return any(kw in prompt.lower() for kw in keywords)

def is_text_prompt(prompt):
    return True  # Always assume some explanation might be needed

async def speak_text(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
    communicate = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
    await communicate.save(tmp_path)
    playsound(tmp_path)
    os.remove(tmp_path)

def handle_text_prompt(prompt):
    payload = {
        "model": "mistral",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(LLM_URL, json=payload)
    if response.status_code == 200:
        try:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                print("\n🤖 Response from LLM:\n", content)
                asyncio.run(speak_text(content))
            else:
                print("No content found in LLM response.")
        except json.JSONDecodeError:
            print("Failed to decode JSON from LLM response.")
    else:
        print(f"Error from LLM server: {response.status_code}")
        print(response.text)

def handle_image_prompt(prompt):
    print("🎨 Generating image...")
    image = hf_client.text_to_image(prompt=prompt, model=HF_MODEL)
    image.show()

# --- Main Agent ---
if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")

    # Determine what actions are required
    generate_image = is_image_prompt(user_prompt)
    generate_text = is_text_prompt(user_prompt)

    # If both image + explanation are in one prompt
    if generate_image and generate_text:
        # Try to split intelligently
        image_part = user_prompt
        explain_part = user_prompt

        # Heuristically split if "and" is in prompt
        if " and " in user_prompt.lower():
            parts = user_prompt.lower().split(" and ", 1)
            image_part = parts[0]
            explain_part = parts[1]

        handle_image_prompt(image_part.strip())
        handle_text_prompt(explain_part.strip())

    elif generate_image:
        handle_image_prompt(user_prompt.strip())

    elif generate_text:
        handle_text_prompt(user_prompt.strip())
