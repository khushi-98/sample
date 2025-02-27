import streamlit as st
import whisper
import torch
import librosa
import soundfile as sf
import noisereduce as nr
from googletrans import Translator
import os

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"🚀 Using device: {device}")

# Load Whisper model (Use 'medium' for better accuracy)
@st.cache_resource  # Caches the model for better performance
def load_model():
    return whisper.load_model("medium").to(device)

model = load_model()

# Initialize Google Translator
translator = Translator()

# Streamlit UI
st.title("🎙️ Whisper Speech-to-Text Transcription")
st.write("Upload an audio file and get its transcription.")

# File Upload
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])

if uploaded_file:
    # Save uploaded file temporarily
    audio_path = "uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ Uploaded file: {uploaded_file.name}")

    # Load and Denoise Audio
    audio, sr = librosa.load(audio_path, sr=16000)
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)

    # Save cleaned audio
    temp_audio_file = "cleaned_audio.wav"
    sf.write(temp_audio_file, reduced_noise, sr)

    # Transcribe Audio
    st.write("⏳ Transcribing...")
    result = model.transcribe(temp_audio_file, language="hi")  # Forced Hindi language
    original_text = result["text"].strip()
    detected_lang = result["language"]

    st.write(f"🌍 Detected Language: {detected_lang} (Forced: Hindi)")
    st.write("📝 Raw Transcription:", original_text if original_text else "[No speech detected]")

    # Translate to English if needed
    translated_text = (
        translator.translate(original_text, src="hi", dest="en").text if detected_lang != "en" else original_text
    )

    st.success("✅ Final Transcription:")
    st.write(translated_text)

    # Cleanup temporary files
    os.remove(audio_path)
    os.remove(temp_audio_file)
