from pyannote.audio import Pipeline
import whisper

# Initialize models
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                use_auth_token="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")
whisper_model = whisper.load_model("medium")

# Process audio
audio_path = r"F:\python\audio_diarization\mini dataset\test\test.wav"
full_audio = whisper.load_audio(audio_path)

diarization = diarization_pipeline(audio_path)

sample_rate = 16000

# Refine diarization with Whisper
results = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    
    start_sample = int(turn.start * sample_rate)
    end_sample = int(turn.end * sample_rate)
    
    # Extract segment
    segment = full_audio[start_sample:end_sample]
    
    # Transcribe
    transcription = whisper_model.transcribe(segment)
    results.append({
        "speaker": speaker,
        "start": turn.start,
        "end": turn.end,
        "text": transcription["text"]
    })


merged = []
for item in results:
    if merged and merged[-1]['speaker'] == item['speaker']:
        merged[-1]['text'] += item['text']
        merged[-1]['end'] = item['end']
    else:
        merged.append(item.copy())

# Print formatted output
for item in merged:
    print(f"{item['speaker']}: {item['text'].strip()}")