import speech_recognition as sr
import pyttsx3
import webbrowser 
import os 
import wikipedia
import pyjokes
import datetime

engine = pyttsx3.init()
engine.setProperty("rate", 100)

def say(text):
    pyttsx3.speak(text)

def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening....")
        r.adjust_for_ambient_noise(source) #recognizer's sensitivity to the ambient noise in the environment
        audio = r.listen(source) #captures the audio input from the microphone source
        
        try:
            print(".......")
            data = r.recognize_google(audio, language="en-IN") #method processes the audio data and tries to convert it into text considering both English and Malayalam languages
            print(data)
            return data.lower()
        except sr.UnknownValueError:
            print("I didn't catch you")

if __name__ == "__main__":
    say("Hello human , HOW CAN I HELP YOU")
    while True:
        r = command()  # Capture user's command
        
        sites = {"google":"https://www.google.com/","spotify":"https://www.spotify.com/","youtube":"https://www.youtube.com/","instagram":"https://www.instagram.com/"}
        for site in sites:
            if f"open {site}" in r:
                say(f"Opening {site.capitalize()}...")
                webbrowser.open(sites[site])
            
        if r == "tell me a joke":
            joke = pyjokes.get_joke()
            print(joke)
            say(joke)    
        elif "tell me about" in r:
            query = r.replace("tell me about", "").strip()
            summary = wikipedia.summary(query, sentences=2)
            print(summary)
            say(summary)
        elif r == "hey" or r == "bro" or r == "hello" or r == "hey":
            print("HOW CAN I HELP YOU")
            say("HOW CAN I HELP YOU")   
            
        elif "the time" in r:
           strfTime = datetime.datetime.now().strftime("%I:%M")
           say(f"the time is {strfTime}")
        elif "tell me the date" in r:
            now = datetime.datetime.now()
            date_str = now.strftime("%A, %B %d, %Y")
            say(f"The date is {date_str}")
            
        elif r == "thank you":
            print("You're welcome!")
            say("You're welcome!")
            break  #"thank you"