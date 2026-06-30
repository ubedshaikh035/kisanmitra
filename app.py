from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import json
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

with open("crop_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

@app.route("/predict", methods=["GET", "POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200
    data = request.get_json(force=True)
    features = [[
        float(data["N"]), float(data["P"]), float(data["K"]),
        float(data["temperature"]), float(data["humidity"]),
        float(data["ph"]), float(data["rainfall"])
    ]]
    prediction = model.predict(features)[0]
    response = jsonify({"crop": prediction})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route("/ai", methods=["GET", "POST", "OPTIONS"])
def ai_endpoint():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    data = request.get_json(force=True)
    task = data.get("task", "")
    context = data.get("context", {})

    prompts = {
        "plant_info": f"""Give farming info about the crop "{context.get('crop', '')}".
Return ONLY valid JSON, no markdown:
{{
  "description": "2-3 sentence description",
  "uses": "2-3 sentence summary of common uses",
  "growing_tips": "3-4 sentences on soil, water, climate",
  "ideal_conditions": "ideal temperature, soil pH, rainfall"
}}""",

        "fertilizer_info": f"""Explain the fertilizer "{context.get('fertilizer', '')}"
recommended for the crop "{context.get('crop', '')}".
Return ONLY valid JSON, no markdown:
{{
  "fertilizer_name": "{context.get('fertilizer', '')}",
  "why_recommended": "2-3 sentences explaining why this fits the crop/soil",
  "application_tips": "2-3 sentences on how/when/how much to apply",
  "warnings": "any precautions or risks of overuse"
}}""",

        "disease_info": f"""Explain the plant disease "{context.get('disease', '')}"
affecting the crop "{context.get('crop', '')}".
Return ONLY valid JSON, no markdown:
{{
  "disease_name": "{context.get('disease', '')}",
  "description": "2-3 sentences about the disease and its symptoms",
  "treatment": "3-4 sentences on how to treat it",
  "prevention": "2-3 sentences on preventing it in future"
}}""",
    }

    prompt = prompts.get(task)
    if not prompt:
        response = jsonify({"error": f"Unknown task: {task}"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 400

    try:
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openrouter/free",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        result = res.json()
        if "choices" not in result:
            raise Exception(f"OpenRouter response: {json.dumps(result)}")
        text =