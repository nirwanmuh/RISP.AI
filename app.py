import streamlit as st
import tempfile
import os
import whisper
from moviepy.editor import VideoFileClip

# Judul aplikasi
st.title("🎙️ Audio/Video to Text Transcriber")

# Upload file
uploaded_file = st.file_uploader("Upload file audio/video (mp3, mp4, wav, m4a...)", type=["mp3", "mp4", "wav", "m4a"])

if uploaded_file is not None:
    # Simpan file ke temporary
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(tmp_path)  # Bisa diputar di Streamlit

    # Jika file video, ekstrak audio
    if uploaded_file.type.startswith("video"):
        st.info("Ekstrak audio dari video...")
        video = VideoFileClip(tmp_path)
        audio_path = tmp_path + ".wav"
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")
    else:
        audio_path = tmp_path

    # Load model whisper
    st.info("Memuat model transkripsi... (butuh waktu awal)")
    model = whisper.load_model("base")  # bisa ganti "small", "medium", "large"

    # Transkripsi
    st.info("Sedang transkripsi audio...")
    result = model.transcribe(audio_path, fp16=False)
    transcript = result["text"]

    # Tampilkan hasil
    st.subheader("📄 Hasil Transkripsi:")
    st.text_area("Teks", transcript, height=300)

    # Tombol download
    st.download_button(
        "💾 Download Transkripsi",
        transcript,
        file_name="transkripsi.txt",
        mime="text/plain"
    )
