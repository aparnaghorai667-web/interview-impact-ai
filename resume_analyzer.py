import json
import os
import requests

# Simple Hugging Face Inference API example.
# Set HUGGINGFACE_API_TOKEN in your environment.

HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"
HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("Set HUGGINGFACE_API_TOKEN in your environment.")

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}


def analyze_resume(resume_text: str, job_description: str) -> dict:
    prompt = (
        "You are an interviewer evaluating resume match to a job. "
        "Compare the resume text with the job description. "
        "Return a JSON object with these fields:\n"
        "  - score: integer from 0 to 100\n"
        "  - missing_skills: list of missing skills mentioned in the job description\n"
        "  - summary: short explanation\n"
        f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}\n\n"
        "Respond only with valid JSON."
    )

    data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.3},
    }

    response = requests.post(HF_API_URL, headers=headers, json=data)
    response.raise_for_status()
    text = response.json()[0]["generated_text"]

    # Try to extract JSON from the model output.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Unable to parse JSON from model output")

    returned_json = text[start : end + 1]
    return json.loads(returned_json)


if __name__ == "__main__":
    sample_resume = "Experienced software engineer with Python, Django, REST APIs, SQL, and cloud deployment experience."
    sample_job = "Looking for a backend engineer with Python, Flask, Docker, AWS, and SQL skills. Experience with microservices is a plus."

    result = analyze_resume(sample_resume, sample_job)
    print(json.dumps(result, indent=2))
