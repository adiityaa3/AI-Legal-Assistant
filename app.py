from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import json
import os
from gtts import gTTS
import tempfile
import base64

app = Flask(__name__)
CORS(app)


API_KEY = "AIzaSyBHHsbe2evrfpg9-MSYnRX3HnXyLHx6KGI"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"


SYSTEM_PROMPT = (
    "You are an AI Legal Assistant specializing in Indian law (IPC and related acts). "
    "Analyze the given crime story and provide a clear explanation in the target language."
) 


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "crimeCategory": {"type": "string"},
        "relevantSection": {"type": "string"},
        "punishmentSummary": {"type": "string"},
        "simplifiedExplanation": {"type": "string"}
    },
    "required": ["crimeCategory", "relevantSection", "punishmentSummary", "simplifiedExplanation"]
}


def analyze_crime_story(story, output_language):
    if not story or len(story) < 2:
        return {"error": "Please provide a detailed story of at least 2 characters."}

    prompt = (
        f"Analyze this crime scenario and explain in {output_language}: '{story}'. "
        "Return only a structured JSON response following the schema."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA
        }
    }

    try:
        response = requests.post(f"{API_URL}?key={API_KEY}", json=payload) #LLMs used
        response.raise_for_status()
        result = response.json()
        json_text = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(json_text)
    except Exception as e:
        return {"error": str(e)}

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    story = data.get("story", "")
    output_language = data.get("output_language", "English")
    result = analyze_crime_story(story, output_language)
    return jsonify(result)


@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                    {"text": "Transcribe this speech accurately and return plain text only."}
                ]
            }
        ]
    }

    try:
        response = requests.post(f"{API_URL}?key={API_KEY}", json=payload) #NLP USED
        response.raise_for_status()
        result = response.json()
        transcription = result["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"transcription": transcription})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/text-to-speech", methods=["POST"])
def text_to_speech():
    data = request.get_json()
    text = data.get("text", "")
    language = data.get("language", "en")

    if not text.strip():
        return jsonify({"error": "Text cannot be empty."}), 400

   
    supported_langs = ["en","hi","bn","ta","te","ml","gu","mr","pa","kn"]
    if language not in supported_langs:
        language = "en"  # fallback to English

    try:
        tts = gTTS(text=text, lang=language) #NLP USED
        temp_path = tempfile.mktemp(suffix=".mp3")
        tts.save(temp_path)
        return send_file(temp_path, mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)