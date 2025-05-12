import os
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai

# Load the environment file
env_path = Path(r"C:\FILLE\ZOFTCARES\projects\chatbots\huggingface\key2.env")  # Use raw string or forward slashes
load_dotenv(dotenv_path=env_path)

# Retrieve the API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found. Check your 'key2.env' file and path.")

# Configure Gemini API
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat()

# Set initial context
chat.send_message(
    "You are a warm, supportive assistant helping students with learning difficulties. "
    "Engage naturally, ask thoughtful follow-up questions, and provide short, encouraging responses."
)

# Initialize speech recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def record_audio():
    """Record audio from the microphone and convert it to text."""
    with sr.Microphone() as source:
        print("🎤 Recording... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        print("Done recording.")

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand. Please try again.")
        return None
    except sr.RequestError:
        print("Error with the recognition service.")
        return None

def text_to_speech(text):
    """Speak the given text aloud."""
    print(f"Gemini: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# Chat loop
print("Gemini: Hello! How can I support you today? (say 'exit' to quit)\n")

while True:
    user_input = record_audio()
    if user_input is None:
        continue

    print(f"👤 You: {user_input}")

    if user_input.lower() in ["exit", "quit"]:
        print("Gemini: Take care! You're doing great.")
        break

    try:
        # Send message to Gemini model and get response
        response = chat.send_message(user_input)
        text_to_speech(response.text.strip())
    except Exception as e:
        print(" Error:", e)
