import streamlit as st
import tempfile
import os
import whisper
import subprocess

st.title("🎙️ Audio/Video to Text Transcriber")

uploaded_file = st.file_uploader(
    "Upload file audio/video (mp3, mp4, wav, m4a...)", 
    type=["mp3", "mp4", "wav", "m4a"]
)

if uploaded_file is not None:
    # Simpan file ke temporary
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(tmp_path)  # Bisa diputar

    # Konversi ke wav (agar whisper lebih stabil)
    audio_path = tmp_path + ".wav"
    subprocess.run([
        "ffmpeg", "-i", tmp_path,
        "-ac", "1", "-ar", "16000",
        audio_path, "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load model whisper
    st.info("Memuat model transkripsi...")
    model = whisper.load_model("base")

    # Transkripsi
    st.info("Sedang transkripsi audio...")
    result = model.transcribe(audio_path, fp16=False)
    transcript = result["text"]

    # Hasil
    st.subheader("📄 Hasil Transkripsi:")
    st.text_area("Teks", transcript, height=300)

    # Tombol download
    st.download_button(
        "💾 Download Transkripsi",
        transcript,
        file_name="transkripsi.txt",
        mime="text/plain"
    )
