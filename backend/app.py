from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# URLs
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
HF_SUMMARY_URL = "https://api-inference.huggingface.co/models/mrm8488/bert2bert_shared-spanish-finetuned-summarization"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No se envió audio"}), 400

    audio_file = request.files['audio']
    
    # --- 1. Transcripción con Groq (Whisper) ---
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    files = {
        'file': ('audio.wav', audio_file.read(), 'audio/wav'),
        'model': (None, 'whisper-large-v3'),
        'language': (None, 'es'),  # Forzar español
        'response_format': (None, 'json')
    }

    try:
        groq_resp = requests.post(GROQ_TRANSCRIBE_URL, headers=headers, files=files)
        groq_resp.raise_for_status()
        transcription = groq_resp.json().get("text", "").strip()
    except Exception as e:
        return jsonify({
            "error": "Fallo en transcripción",
            "details": str(e),
            "raw": groq_resp.text if 'groq_resp' in locals() else "No response"
        }), 500

    if not transcription:
        return jsonify({"error": "Transcripción vacía"}), 400

    # --- 2. Resumen con Hugging Face ---
    hf_headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    hf_payload = {
        "inputs": transcription[:1000],
        "parameters": {"max_length": 100, "min_length": 20, "do_sample": False}
    }

    try:
        hf_resp = requests.post(HF_SUMMARY_URL, headers=hf_headers, json=hf_payload, timeout=30)
        if hf_resp.status_code == 200:
            summary = hf_resp.json()[0].get("generated_text", "Resumen no disponible.")
        else:
            # Si el modelo está "cold", HF devuelve error 503 – manejamos con mensaje amable
            summary = "Resumen temporalmente no disponible. Inténtalo de nuevo en 10 segundos."
    except Exception as e:
        summary = f"Error al generar resumen: {str(e)}"

    # --- 3. Extracción simple de tareas (mejorable, pero funcional) ---
    task_keywords = ["debo", "tengo que", "necesito", "recordar", "hacer", "enviar", "revisar", "preparar", "agendar"]
    sentences = [s.strip() + "." for s in transcription.split(".") if s.strip()]
    tasks = [
        sent for sent in sentences
        if any(kw in sent.lower() for kw in task_keywords)
    ] or ["No se detectaron tareas pendientes."]

    return jsonify({
        "transcription": transcription,
        "summary": summary,
        "tasks": tasks
    })

# ✅ Solo para pruebas locales
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # ¡debug=False en producción!