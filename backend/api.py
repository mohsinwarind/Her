# from openai import OpenAI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import sounddevice as sd
import scipy.io.wavfile as wav
import pyttsx3
engine = pyttsx3.init() # object creation

import os
load_dotenv() #loading dot env file

client = InferenceClient(token=os.getenv("HF_TOKEN"))
selected_model = "openai/gpt-oss-120b"
def speak(text):
    tts = pyttsx3.init()
    tts.say(text)
    tts.runAndWait()
    tts.stop()
print("Loading model...")
model = WhisperModel("small", device="cpu", compute_type="float32")
print("Model loaded.")
fs = 16000
print("Recording audio for 5 seconds...")
while True:
    print("Recording...")

    recording = sd.rec(
        int(5 * fs),
        samplerate=fs,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    audio = recording.squeeze().astype("float32") / 32768.0

    print("Transcribing...")

    segments, info = model.transcribe(
        audio,
        language="en",
        vad_filter=True
    )
    segments = list(segments)
    print("Segments:", segments)
    userquery = "You are a helpful assistant. Answer the following question in detail: "

    for segment in segments:
        userquery += segment.text + " "
        # userquery = " ".join(segment.text for segment in segments).strip()
        print(segment.text)
    print("Initiating response generation...")
    response = client.chat_completion(
    messages=[{"role": "user", "content": userquery}],
    model=selected_model,
    max_tokens=100,
    )

    speak(response.choices[0].message.content)
    print(response.choices[0].message.content)
        
