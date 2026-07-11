from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from google import genai
from google.genai import types
import os
import json
from gtts import gTTS
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ---------------- #

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

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
        return {
            "error": "Please provide a valid crime story."
        }

    prompt = f"""
{SYSTEM_PROMPT}

Crime Story:
{story}

Respond in {output_language}.

Return ONLY valid JSON with this structure:

{{
    "crimeCategory":"",
    "relevantSection":"",
    "punishmentSummary":"",
    "simplifiedExplanation":""
}}
"""
try:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {
            "error": "Gemini returned invalid JSON",
            "response": response.text
        }

except Exception as e:
    return {
        "error": str(e)
    }
@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    story = data.get("story", "")
    output_language = data.get("output_language", "English")

    result = analyze_crime_story(story, output_language)

    return jsonify(result)
# ---------------- SPEECH TO TEXT ---------------- #

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():

    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

 audio = request.files["audio"]

audio_bytes = audio.read()
mime_type = audio.mimetype or "audio/wav"

try:

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type
            ),
            "Transcribe this speech accurately. Return plain text only."
        ]
    )

        return jsonify({
            "transcription": response.text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

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
