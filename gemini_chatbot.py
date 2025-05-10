import google.generativeai as genai

# Replace with your actual Gemini API key
GEMINI_API_KEY = "AIzaSyBsAedEP_qBERNTcIDx1p_HD9-kT7ynwFY"

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Attempt to initialize the Gemini Pro model
try:
    model = genai.GenerativeModel('models/gemini-pro')
    print("Gemini Pro model initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini Pro model: {e}")
    print("Attempting to list available models to find a valid one...")
    available_models = []
    try:
        for model_info in genai.list_models():
            print(f"Available Model: {model_info.name}")
            print(f"  Supported Methods: {model_info.supported_generation_methods}")
            available_models.append(model_info.name)
            print("-" * 20)
        if available_models:
            print("\nPlease try using one of the models listed above in the 'genai.GenerativeModel()' line.")
            print("For example: genai.GenerativeModel('the_name_of_a_model_from_the_list')")
        else:
            print("No models were found. Please check your API key and internet connection.")
        exit()

def chat_with_gemini():
    """A simple chatbot that interacts with the Gemini model."""
    print("\nHello! I'm a simple chatbot powered by Gemini. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            response = model.generate_content(user_input)
            print(f"Bot: {response.text}")
        except Exception as e:
            print(f"An error occurred during chat: {e}")
            print("Please check your input or try again later.")

if __name__ == "__main__":
    chat_with_gemini()