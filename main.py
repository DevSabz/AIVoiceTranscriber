from openai import OpenAI
import sounddevice as sd
import numpy as np
from scipy.io import wavfile  # Use scipy for handling WAV files
import os
from tqdm import tqdm  # Import tqdm for the progress bar
import time

# Initialize OpenAI API client
gpt = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Function for recording audio
def record_audio(filename="output.wav", duration=0, fs=44100, channels=1):

    # Overwrite existing file if it exists
    if filename is not None:
        try:
            os.remove(filename)
        except OSError:
            pass

    # Start recording
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()

    # Show recording progressbar # _ used as a variable because it wont be used again. 
    #for _ in tqdm(range(duration), desc="Recording", unit ="s"):
    #   time.sleep(1)

    # Save recording to file using scipy.io.wavfile
    myrecording = np.int16(myrecording * 32767)  # Convert from float32 to int16
    wavfile.write(filename, fs, myrecording)


# Prompt user for recording duration
recording_duration = int(input("Enter recording duration in seconds: "))

# Record audio
if recording_duration > 0:
    # Record for a specific duration
    record_audio("recording.wav", duration=recording_duration)
else:
    # Continuous recording (stop with Ctrl+C)
    record_audio("recording.wav")

# Open the recorded audio file
audio_file = open("recording.wav", "rb")

# Transcribe audio using OpenAI's Whisper model
transcription = gpt.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)

# Print the transcription
print()
print(transcription.text)
print()
