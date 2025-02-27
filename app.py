import streamlit as st
import whisper
import torch
import librosa
import soundfile as sf
import noisereduce as nr
import os

# Load Whisper Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium").to(device)

# Streamlit UI
st.title("🎙️ Whisper Speech-to-Text Transcription")
st.write("Upload an audio file and get the transcribed text.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    file_path = "temp_audio.wav"
    
    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format="audio/wav")

    # Load and Denoise Audio
    audio, sr = librosa.load(file_path, sr=16000)
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
    sf.write(file_path, reduced_noise, sr)

    # Transcribe using Whisper
    with st.spinner("Transcribing... ⏳"):
        result = model.transcribe(file_path)
        transcription = result["text"].strip()
        detected_lang = result["language"]

    st.success("✅ Transcription Complete!")
    st.subheader("📝 Transcribed Text:")
    st.write(transcription)

    # Save transcription to a text file
    txt_filename = "transcription.txt"
    with open(txt_filename, "w") as f:
        f.write(transcription)

    st.download_button(label="📥 Download Transcription", data=transcription, file_name=txt_filename)

    # Cleanup temporary files
    os.remove(file_path)
