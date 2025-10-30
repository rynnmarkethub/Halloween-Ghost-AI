import os
import speech_recognition as sr
import google.generativeai as genai
import threading
import time
import pyttsx3

# Gemini setup
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Error: GENAI_API_KEY not set.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')
chat_session = model.start_chat(history=[
    {"role": "user", "parts": "From now on, you are a ghost haunting a house on Halloween night. "
                              "You have returned from the dead to greet trick-or-treaters at your doorstep. "
                              "Speak in a threatening, dramatic, and haunting tone, with a deep, gravelly, "
                              "ghost-like voice. Keep your responses short, 1–2 sentences, and make them chilling, "
                              "mysterious, and playful enough to frighten or intrigue children. "
                              "Use spooky Halloween imagery and sudden dramatic pauses to enhance the eerie effect."}
])

# Speech recognition
r = sr.Recognizer()

def speak(text):
    # Initialize pyttsx3 inside a thread each time to prevent locking
    def run_tts():
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[0].id)
            engine.setProperty('rate', 140)
            engine.setProperty('volume', 1.0)
            print(f"Bot: {text}")
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

    t = threading.Thread(target=run_tts)
    t.start()
    time.sleep(0.1)

def listen_for_speech():
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, phrase_time_limit=5)
    try:
        command = r.recognize_google(audio)
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
        return ""
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return ""

def main():
    speak("What is it you want, mortal?")
    while True:
        user_input = listen_for_speech()
        if user_input:
            if any(word in user_input.lower() for word in ["stop", "goodbye", "bye", "thank you"]):
                speak("Until next time, mortal.")
                break
            try:
                response = chat_session.send_message(user_input)
                speak(response.text)
            except Exception as e:
                speak("What are you rambling on about, mortal?")
                print(f"LLM Error: {e}")

if __name__ == "__main__":
    main()
