from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")

# URLs
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
HF_SUMMARY_URL = "https://api-inference.huggingface.co/models/mrm8488/bert2bert_shared-spanish-finetuned-summarization"
EXCHANGE_RATE_URL = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/USD"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# ====== TU ENDPOINT ORIGINAL (/transcribe) ======
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No se envió audio"}), 400

    audio_file = request.files['audio']
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    files = {
        'file': ('audio.wav', audio_file.read(), 'audio/wav'),
        'model': (None, 'whisper-large-v3'),
        'language': (None, 'es'),
        'response_format': (None, 'json')
    }

    try:
        groq_resp = requests.post(GROQ_TRANSCRIBE_URL, headers=headers, files=files)
        groq_resp.raise_for_status()
        transcription = groq_resp.json().get("text", "").strip()
    except Exception as e:
        return jsonify({
            "error": "Fallo en transcripción",
            "details": str(e)
        }), 500

    if not transcription:
        return jsonify({"error": "Transcripción vacía"}), 400

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
            summary = "Resumen temporalmente no disponible. Inténtalo de nuevo en 10 segundos."
    except Exception:
        summary = "Error al generar resumen."

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

# ====== NUEVO: /command ======
@app.route('/command', methods=['POST'])
def command():
    texto = request.json.get("text", "").strip().lower()
    if not texto:
        return jsonify({"error": "Texto vacío"}), 400

    # === 1. Detección de intención ===
    if "clima" in texto or "tiempo" in texto:
        # Extraer ciudad (por ahora solo Guatemala como fallback)
        ciudad = "Guatemala"
        if "antigua" in texto:
            ciudad = "Antigua Guatemala"
        elif "ciudad" in texto and "guatemala" in texto:
            ciudad = "Ciudad de Guatemala"
        # Puedes mejorar esto con NER o regex después

        # Usamos coordenadas fijas (puedes mejorar con geocoding después)
        coords = {
            "Guatemala": (14.6407, -90.5133),
            "Antigua Guatemala": (14.5589, -90.7308),
            "Ciudad de Guatemala": (14.6407, -90.5133)
        }.get(ciudad, (14.6407, -90.5133))

        lat, lon = coords
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "timezone": "America/Guatemala"
        }
        try:
            resp = requests.get(OPEN_METEO_URL, params=params)
            data = resp.json()
            temp = data["current"]["temperature_2m"]
            wcode = data["current"]["weather_code"]
            # Código simple de clima (ver: https://open-meteo.com/en/docs)
            clima_desc = {
                0: "despejado", 1: "principalmente despejado", 2: "parcialmente nublado",
                3: "nublado", 45: "niebla", 48: "niebla con escarcha",
                51: "lluvia ligera", 53: "lluvia moderada", 55: "lluvia intensa",
                61: "llovizna", 63: "lluvia", 65: "lluvia fuerte",
                71: "nevada ligera", 73: "nevada moderada", 75: "nevada fuerte",
                95: "tormenta", 96: "tormenta con granizo ligero", 99: "tormenta con granizo fuerte"
            }.get(wcode, "condiciones desconocidas")
            return jsonify({"response": f"Hoy en {ciudad}: {temp}°C, {clima_desc}."})
        except Exception as e:
            return jsonify({"response": f"No pude obtener el clima. Error: {str(e)}"})

    elif "dólar" in texto or "cambio" in texto or "tipo de cambio" in texto:
        if not EXCHANGE_RATE_API_KEY:
            return jsonify({"response": "No está configurada la clave para tipo de cambio."})
        try:
            resp = requests.get(EXCHANGE_RATE_URL)
            data = resp.json()
            if data.get("result") != "success":
                return jsonify({"response": "Error en la API de tipo de cambio."})
            gtq = data["conversion_rates"].get("GTQ")
            vef = data["conversion_rates"].get("VEF")  # Nota: VEF está obsoleto, pero BCV/VE tiene su propia API
            if gtq:
                return jsonify({"response": f"El dólar está a {gtq:.2f} quetzales."})
            else:
                return jsonify({"response": "No pude obtener el tipo de cambio para tu moneda."})
        except Exception as e:
            return jsonify({"response": f"No pude obtener el tipo de cambio. Error: {str(e)}"})

    else:
        return jsonify({"response": "Lo siento, no entendí ese comando. Prueba: 'clima en Antigua' o '¿a cuánto está el dólar?'."})

# === Para pruebas locales ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
