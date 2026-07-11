from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import json
import os
from gtts import gTTS
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ---------------- #

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

MODEL_NAME = "gemini-2.5-flash"
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
    "required": [
        "crimeCategory",
        "relevantSection",
        "punishmentSummary",
        "simplifiedExplanation"
    ]
}

# ---------------- HOME ---------------- #

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "AI Legal Assistant API is live"
    })

# ---------------- ANALYZE ---------------- #

def analyze_crime_story(story, output_language):

    if not story or len(story.strip()) < 2:
        return {"error": "Please provide a valid crime story."}

    prompt = (
        f"Analyze this crime scenario and explain in {output_language}: '{story}'. "
        "Return ONLY valid JSON following the provided schema."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "systemInstruction": {
            "parts": [
                {"text": SYSTEM_PROMPT}
            ]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA
        }
    }

    try:

        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        json_text = result["candidates"][0]["content"]["parts"][0]["text"]

        return json.loads(json_text)

    except requests.exceptions.RequestException as e:
        return {"error": f"Gemini API Error: {str(e)}"}

    except Exception as e:
        return {"error": f"Server Error: {str(e)}"}


@app.route("/analyze", methods=["POST"])
def analyze():
    print("Analyze endpoint called")

    data = request.get_json()
    print(data)

    story = data.get("story", "")
    output_language = data.get("output_language", "English")

    result = analyze_crime_story(story, output_language)

    print(result)

    return jsonify(result)

# ---------------- SPEECH TO TEXT ---------------- #

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():

    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio = request.files["audio"]

    audio_b64 = base64.b64encode(audio.read()).decode()

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": audio_b64
                        }
                    },
                    {
                        "text": "Transcribe this speech accurately and return only plain text."
                    }
                ]
            }
        ]
    }

    try:

        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        text = result["candidates"][0]["content"]["parts"][0]["text"]

        return jsonify({"transcription": text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

# ---------------- TEXT TO SPEECH ---------------- #

@app.route("/text-to-speech", methods=["POST"])
def text_to_speech():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    text = data.get("text", "")

    language = data.get("language", "en")[:2]

    if not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400

    supported = [
        "en",
        "hi",
        "bn",
        "ta",
        "te",
        "ml",
        "gu",
        "mr",
        "pa",
        "kn"
    ]

    if language not in supported:
        language = "en"

    try:

        tts = gTTS(text=text, lang=language)

        audio = BytesIO()

        tts.write_to_fp(audio)

        audio.seek(0)

        return send_file(
            audio,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="speech.mp3"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
