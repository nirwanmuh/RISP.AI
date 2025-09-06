# streamlit_realtime_mom.py
# Streamlit app: Real-time / file audio -> transcription -> LLM (Gemini) correction -> generate MOM
# Features:
# - Upload audio file (wav, mp3) OR record from microphone (requires streamlit-webrtc)
# - Real-time-ish transcription using local whisper (if installed) or SpeechRecognition fallback
# - Text correction via Gemini API (Google Generative AI)
# - Generate Minutes of Meeting (MOM) with sections: Topic & Kesepakatan

import streamlit as st
from pathlib import Path
import tempfile
import os
import json
from typing import List, Dict

# Gemini integration
import google.generativeai as genai

# Configure Gemini API (API key must be set in Streamlit secrets)
if "gemini_api_key" in st.secrets:
    genai.configure(api_key=st.secrets["gemini_api_key"])

# Optional imports ---------------------------------------------------------
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
    WEBSRTC_AVAILABLE = True
except Exception:
    WEBSRTC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Utility helpers

def save_uploaded_file(uploaded_file) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1])
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def transcribe_with_whisper_local(audio_path: str, model_name: str = "small") -> str:
    if not WHISPER_AVAILABLE:
        raise RuntimeError("whisper not installed. Install via `pip install -U openai-whisper`.")

    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except FileNotFoundError as e:
        # Biasanya karena ffmpeg tidak tersedia
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source_audio:
            audio_data = r.record(source_audio)
        try:
            return r.recognize_google(audio_data, language='id-ID')
        except Exception as e2:
            return f"[Error fallback transcription: {e2}]"


# Gemini LLM correction and structuring

def call_gemini_correct_and_structure(raw_text: str, api_config: Dict = None) -> Dict:
    if not raw_text.strip():
        return {"corrected": "", "topics": []}

    prompt = f"""
    Anda adalah asisten yang membantu membuat notulen rapat (Minutes of Meeting).
    Berikut adalah transkrip rapat:

    ---
    {raw_text}
    ---

    Tugas Anda:
    1. Perbaiki teks agar lebih rapi dan sesuai kaidah bahasa.
    2. Ringkas menjadi notulen dengan format JSON:
       {{
         "corrected": "(teks transkrip yang sudah diperbaiki)",
         "topics": [
           {{"topic": "Judul Topik", "kesepakatan": "Isi kesepakatan"}},
           ...
         ]
       }}

    Kembalikan hanya JSON valid tanpa penjelasan tambahan.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    text_response = response.text.strip()

    try:
        parsed = json.loads(text_response)
    except Exception:
        parsed = {"corrected": raw_text, "topics": [{"topic": "Umum", "kesepakatan": text_response}]}

    return parsed


# Render MOM as markdown

def render_mom(topics: List[Dict]) -> str:
    md = "# Minutes of Meeting\n\n"
    for idx, t in enumerate(topics, start=1):
        md += f"## {idx}. {t.get('topic','(no topic)')}\n\n"
        md += f"**Kesepakatan:** {t.get('kesepakatan','-')}\n\n"
    return md


# App UI -------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Realtime MOM - Streamlit", layout='wide')
    st.title("Realtime / File Audio → Transcription → Gemini → MOM")

    st.sidebar.header("Sumber Audio")
    source = st.sidebar.radio("Pilih sumber audio:", ["Upload File", "Microphone (webrtc)"] if WEBSRTC_AVAILABLE else ["Upload File"], index=0)

    st.sidebar.header("Transcription Engine")
    use_whisper_local = st.sidebar.checkbox("Gunakan Whisper lokal (jika tersedia)", value=WHISPER_AVAILABLE)
    whisper_model = st.sidebar.selectbox("Whisper model (lokal)", ["tiny", "base", "small", "medium", "large"], index=2)

    # Main columns
    col1, col2 = st.columns([2,1])

    with col1:
        st.header("Transcription / Live text")

        transcript_box = st.empty()
        raw_transcript = st.session_state.get('raw_transcript', '')

        if source == "Upload File":
            uploaded = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a","ogg"])
            if uploaded is not None:
                filepath = save_uploaded_file(uploaded)
                st.info(f"Saved to {filepath}")
                with st.spinner("Transcribing..."):
                    if use_whisper_local and WHISPER_AVAILABLE:
                        text = transcribe_with_whisper_local(filepath, model_name=whisper_model)
                    else:
                        try:
                            import speech_recognition as sr
                            r = sr.Recognizer()
                            with sr.AudioFile(filepath) as source_audio:
                                audio_data = r.record(source_audio)
                            text = r.recognize_google(audio_data, language='id-ID')
                        except Exception as e:
                            text = f"[Error: no transcription engine available: {e} ]"
                    raw_transcript = text
                    st.session_state['raw_transcript'] = raw_transcript
                    transcript_box.code(raw_transcript)

        elif source == "Microphone (webrtc)":
            st.info("Microphone capture via streamlit-webrtc. Click Start to begin.\nTranscription will run when you press 'Transcribe last recorded segment'.")
            webrtc_ctx = webrtc_streamer(key="mrtc", mode=WebRtcMode.SENDONLY, client_settings=ClientSettings(media_stream_constraints={"audio": True, "video": False}))
            if webrtc_ctx.state.playing:
                st.write("Recording... click Stop to end stream and then click 'Transcribe last recorded segment'")
            if st.button("Transcribe last recorded segment"):
                st.warning("Recording segments are not saved automatically in this demo. Prefer Upload File for now.")

        st.markdown("---")
        st.subheader("Raw Transcript")
        st.text_area("transcript", value=st.session_state.get('raw_transcript',''), height=200, key='ta_transcript')

        if st.button("Perbaiki & Strukturkan (Gemini)"):
            raw_text = st.session_state.get('raw_transcript','')
            if not raw_text.strip():
                st.error("Tidak ada teks untuk diperbaiki. Upload file atau rekam audio terlebih dahulu.")
            else:
                with st.spinner("Mengirim ke Gemini untuk perbaikan dan struktur..."):
                    llm_result = call_gemini_correct_and_structure(raw_text)
                    st.session_state['llm_result'] = llm_result
                    st.success("Selesai")

        if st.session_state.get('llm_result'):
            st.subheader("Hasil Perbaikan (Gemini)")
            st.write(st.session_state['llm_result'].get('corrected',''))

    with col2:
        st.header("Generate MOM")
        if st.session_state.get('llm_result'):
            topics = st.session_state['llm_result'].get('topics', [])
            st.subheader("Topik & Kesepakatan (sunting jika perlu)")
            edited_topics = []
            for i, t in enumerate(topics):
                st.markdown(f"**Topik {i+1}**")
                top = st.text_input(f"Topik {i+1}", value=t.get('topic',''), key=f'topic_{i}')
                kep = st.text_area(f"Kesepakatan {i+1}", value=t.get('kesepakatan',''), key=f'kes_{i}')
                edited_topics.append({"topic": top, "kesepakatan": kep})

            if st.button("Generate MOM Markdown"):
                mom_md = render_mom(edited_topics)
                st.session_state['mom_md'] = mom_md
                st.markdown("MOM generated below — you can copy or download it as a .md file")

        else:
            st.info("Belum ada hasil Gemini. Tekan 'Perbaiki & Strukturkan' dulu setelah transkripsi selesai.")

        if st.session_state.get('mom_md'):
            st.download_button("Download MOM (.md)", data=st.session_state['mom_md'], file_name='MOM.md', mime='text/markdown')
            st.markdown(st.session_state['mom_md'])

    st.sidebar.markdown("---")
    st.sidebar.header("Petunjuk & Dependensi")
    st.sidebar.markdown(
        """
        Dependensi yang diperlukan:
        - streamlit
        - openai-whisper (opsional, untuk transkripsi lokal)
        - streamlit-webrtc (opsional, untuk perekaman mic di browser)
        - speechrecognition (fallback untuk file)
        - google-generativeai (untuk Gemini)

        Cara menjalankan:
        1. Simpan API Key Gemini Anda di `.streamlit/secrets.toml`:
           ```toml
           gemini_api_key = "YOUR_API_KEY"
           ```
        2. Pasang dependensi: `pip install streamlit openai-whisper speechrecognition streamlit-webrtc google-generativeai`
        3. Jalankan: `streamlit run streamlit_realtime_mom.py`
        """
    )

if __name__ == '__main__':
    if 'raw_transcript' not in st.session_state:
        st.session_state['raw_transcript'] = ''
    main()
