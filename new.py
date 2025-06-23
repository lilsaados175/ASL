import sounddevice as sd
import wavio
import numpy as np
import speech_recognition as sr
import pyttsx3

DURATION = 5  # seconds
SAMPLE_RATE = 44100

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def record_audio(filename="temp.wav"):
    print("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    wavio.write(filename, audio, SAMPLE_RATE, sampwidth=2)
    print("Recording complete.")

def recognize_audio(filename="temp.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        speak("Sorry, I didn't understand.")
    except sr.RequestError:
        print("Could not request results.")
        speak("Error connecting to speech service.")
    return ""

def main():
    while True:
        speak("Listening for a command.")
        record_audio()
        command = recognize_audio()

        if "hello" in command:
            speak("Hello there!")
        elif "exit" in command or "stop" in command:
            speak("Goodbye!")
            break
        elif "project" in command:
            speak("Running your project.")
            # run_your_project()
        else:
            speak("Command not recognized.")

if __name__ == "__main__":
    main()
