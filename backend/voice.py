from faster_whisper import WhisperModel
import sounddevice as sd
import scipy.io.wavfile as wav

print("Loading model...")
model = WhisperModel("small", device="cpu", compute_type="int8")
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

    wav.write("recording.wav", fs, recording)

    print("Transcribing...")

    segments, info = model.transcribe(
        "recording.wav",
        vad_filter=False
    )

    for segment in segments:
        print(segment.text)