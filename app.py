import streamlit as st
import tempfile
import os
from faster_whisper import WhisperModel

st.title("🎙️ Audio to Text Transcriber")

uploaded_file = st.file_uploader(
    "Upload file audio (mp3, wav, m4a)", 
    type=["mp3", "wav", "m4a"]
)

if uploaded_file is not None:
    # Simpan file ke temporary
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(tmp_path)

    # Load faster-whisper
    st.info("Memuat model transkripsi...")
    model = WhisperModel("base", device="cpu", compute_type="int8")

    # Transkripsi
    st.info("Sedang transkripsi audio...")
    segments, info = model.transcribe(tmp_path)

    transcript = " ".join([seg.text for seg in segments])

    # Hasil
    st.subheader("📄 Hasil Transkripsi:")
    st.text_area("Teks", transcript.strip(), height=300)

    # Tombol download
    st.download_button(
        "💾 Download Transkripsi",
        transcript.strip(),
        file_name="transkripsi.txt",
        mime="text/plain"
    )
