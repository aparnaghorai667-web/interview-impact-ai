import streamlit as st
import openai
import os

# 1. Setup API (Ensure you have your key set in your environment)
client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")

# 2. Define your functions
def transcribe_audio(audio_file):
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text

def analyze_answer(transcript, job_desc):
    prompt = f"Evaluate this interview answer against the job description: {job_desc}. Answer: {transcript}. Provide a score (0-100) and feedback in JSON format."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# 3. Create the Web Interface
st.title("AI Interview Coach")
job_desc = st.text_area("Paste Job Description")
audio_file = st.file_uploader("Upload your audio response", type=["mp3", "wav"])

if st.button("Evaluate"):
    if audio_file and job_desc:
        with st.spinner("Processing your interview..."):
            # The "Glue" Logic:
            text = transcribe_audio(audio_file)
            result = analyze_answer(text, job_desc)
            
            st.write("### Evaluation Results")
            st.json(result)
    else:
        st.error("Please provide both a job description and an audio file.")