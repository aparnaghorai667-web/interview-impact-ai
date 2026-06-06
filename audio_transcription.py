import os
import openai

# Simple OpenAI Whisper transcription script.
# Set OPENAI_API_KEY in your environment.

openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Set OPENAI_API_KEY in your environment.")


def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcript["text"]


if __name__ == "__main__":
    audio_path = "sample_audio.mp3"  # change to your audio file path
    text = transcribe_audio(audio_path)
    print("Transcribed text:\n")
    print(text)
