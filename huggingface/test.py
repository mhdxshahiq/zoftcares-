import os
from dotenv import load_dotenv
import pyttsx3
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from key.env file
load_dotenv(r"C:\FILLE\ZOFTCARES\projects\chatbots\huggingface\key.env")

# Get the API token from environment variables
sec_key = os.getenv('HF_TOKEN')

# Check if the API key is found
if not sec_key:
    print("Error: API key not found in .env file.")
    exit(1)

# Initialize Hugging Face endpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=sec_key,
    temperature=0.7,
    max_new_tokens=128
)

# Generate a response from the model (use `invoke` instead of `invoke_conversational`)
response = llm.invoke("what is earth")  # This is the correct method

# Initialize the pyttsx3 engine for text-to-speech
engine = pyttsx3.init()

# Check if the response is a valid string
if isinstance(response, str):
    print("🤖 Response:", response)  # Print response to console
    engine.say(response)
    engine.runAndWait()
else:
    print("Error: Response is not a valid string.")
