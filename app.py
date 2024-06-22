import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, ClientSettings
import numpy as np
from scipy.io.wavfile import write
from transformers import pipeline
import nemo.collections.asr as nemo_asr
from tempfile import NamedTemporaryFile

# Initialize ASR model
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# Initialize emotion recognition model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True)

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = []

    def recv(self, frame):
        audio_frame = frame.to_ndarray()
        self.audio_data.append(audio_frame)
        return frame

def transcribe_audio(audio_path):
    transcript = asr_model.transcribe([audio_path])[0]
    return transcript

def recognize_emotion(text):
    emotion_predictions = emotion_classifier(text)
    return emotion_predictions

st.title("Speech Emotion Recognition")

# WebRTC Streamer for audio recording
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)

if st.button("Process Audio"):
    if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
        audio_processor = webrtc_ctx.audio_processor
        audio_data = np.concatenate(audio_processor.audio_data)

        # Save audio data to a temporary file
        temp_audio = NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_audio.name, 16000, audio_data.astype(np.int16))

        st.audio(temp_audio.name, format="audio/wav")

        # Transcription
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(temp_audio.name)
            st.write("Transcript:")
            st.write(transcript)

        # Emotion Recognition
        with st.spinner("Recognizing emotion..."):
            emotion_predictions = recognize_emotion(transcript)
            st.write("Emotion Predictions:")
            for emotion in emotion_predictions[0]:
                st.write(f"{emotion['label']}: {emotion['score']:.4f}")

# File uploader
st.header("Upload an audio file")
uploaded_file = st.file_uploader("Choose a file", type=["wav"])

if uploaded_file is not None:
    with st.spinner("Transcribing audio..."):
        audio_path = uploaded_file.name
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(audio_path, format="audio/wav")

        # Transcription
        transcript = transcribe_audio(audio_path)
        st.write("Transcript:")
        st.write(transcript)

        # Emotion Recognition
        with st.spinner("Recognizing emotion..."):
            emotion_predictions = recognize_emotion(transcript)
            st.write("Emotion Predictions:")
            for emotion in emotion_predictions[0]:
                st.write(f"{emotion['label']}: {emotion['score']:.4f}")
