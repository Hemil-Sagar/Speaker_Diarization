import streamlit as st
from pyannote.audio import Pipeline
import whisper
import tempfile
import os
import warnings
import soundfile as sf

# Streamlit UI settings
st.set_page_config(page_title="Speaker Diarization App", page_icon="🎙️", layout="wide")
st.title("🎙️ Speaker Diarization + Transcription")

warnings.filterwarnings("ignore")

# Hugging Face token (you can put in a secure way later)
HF_TOKEN = "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"

# Load models with cache
@st.cache_resource(show_spinner=True)
def load_models():
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    whisper_model = whisper.load_model("medium")
    return diarization_pipeline, whisper_model

diarization_pipeline, whisper_model = load_models()
sample_rate = 16000

# File upload UI
uploaded_file = st.file_uploader("📁 Upload an audio file", type=["wav", "mp3", "ogg", "m4a", "flac"])
if not uploaded_file:
    st.info("Upload an audio file to get started.")
    st.stop()

# Save the uploaded file temporarily
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    tmp.write(uploaded_file.read())
    audio_path = tmp.name

# Button to process
if st.button("✨ Process Audio"):
    with st.spinner("Processing..."):
        try:
            # Load and run diarization
            full_audio = whisper.load_audio(audio_path)
            diarization = diarization_pipeline(audio_path)

            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_sample = int(turn.start * sample_rate)
                end_sample = int(turn.end * sample_rate)
                segment = full_audio[start_sample:end_sample]

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as segment_file:
                    sf.write(segment_file.name, segment, sample_rate)
                    transcription = whisper_model.transcribe(segment_file.name)

                results.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "text": transcription["text"]
                })

            # Merge consecutive same-speaker segments
            merged = []
            for item in results:
                if merged and merged[-1]['speaker'] == item['speaker']:
                    merged[-1]['text'] += " " + item['text']
                    merged[-1]['end'] = item['end']
                else:
                    merged.append(item.copy())

            # Display results
            if not merged:
                st.warning("❌ No speech segments found.")
            else:
                speakers = sorted(set(x['speaker'] for x in merged))
                st.success(f"✅ Found {len(speakers)} speaker(s)")
                tabs = st.tabs([f"👤 {sp.split('_')[-1]}" for sp in speakers])
                for i, speaker in enumerate(speakers):
                    with tabs[i]:
                        for item in [x for x in merged if x['speaker'] == speaker]:
                            st.markdown(f"""
                            **⏱ {item['start']:.1f}s - {item['end']:.1f}s**  
                            {item['text'].strip()}  
                            ---
                            """)

        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

# Sidebar instructions
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. Upload an audio file (preferably clear recording with 2+ speakers).
    2. Click **Process Audio**.
    3. See transcriptions grouped by speaker.

    **Tips:**  
    - Use `.wav` or `.flac` for best quality  
    - Clear speech helps recognition  
    - First run might take time due to model loading
    """)
