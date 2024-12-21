import os
import warnings
import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from dotenv import load_dotenv
from groq import Groq
import torch

# Load environment variables from .env file
load_dotenv()
warnings.filterwarnings("ignore")

# Load RoBERTa model and tokenizer from Hugging Face
MODEL_NAME = "yourfavouritedotcom/roberta-personality-prediction"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)

# Streamlit app
st.title("Personality Prediction from Audio")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    # Initialize Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Groq API key not found in environment variables.")
    else:
        client = Groq(api_key=groq_api_key)

        # Process transcription
        with st.spinner("Processing audio transcription..."):
            transcription = client.audio.transcriptions.create(
                file=("uploaded_audio.wav", uploaded_file.read()),
                model="whisper-large-v3",
                temperature=0,
                response_format="verbose_json",
            )

        # Display transcription
        st.subheader("Transcription")
        for segment in transcription.segments:
            st.write(f"Segment {segment['id']}: {segment['text']}")

        # Personality analysis
        st.subheader("Personality Analysis")
        personality_analysis = []
        for segment in transcription.segments:
            text = segment["text"]

            # Tokenize input text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            # Predict personality traits
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)

            # Extract the most probable personality type
            predicted_class = torch.argmax(probabilities, dim=1).item()
            analysis = {
                "text": text,
                "predicted_class": predicted_class,
                "probabilities": probabilities.tolist()
            }
            personality_analysis.append(analysis)

            # Display the analysis
            st.write(f"Segment: {text}")
            st.write(f"Predicted Personality Class: {predicted_class}")
            st.write(f"Probabilities: {probabilities.tolist()}")

