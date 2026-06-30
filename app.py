from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import json
import google.generativeai as genai

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

with open("crop_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = gemini_model = genai.GenerativeModel("gemini-2.0-flash")

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

@app.route("/plant-info", methods=["GET", "POST", "OPTIONS"])
def plant_info():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    data = request.get_json(force=True)
    crop_name = data.get("crop", "")

    prompt = f"""Give me information about growing the crop "{crop_name}" as a farmer would need.
Return ONLY valid JSON, no markdown, no backticks, in exactly this format:
{{
  "description": "2-3 sentence description of the crop",
  "uses": "2-3 sentence summary of common uses",
  "growing_tips": "3-4 sentence practical growing tips covering soil, water, and climate",
  "ideal_conditions": "short summary of ideal temperature, soil pH, and rainfall"
}}"""

    try:
        result = gemini_model.generate_content(prompt)
        text = result.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        info = json.loads(text)
    except Exception as e:
        info = {
            "description": "Information unavailable right now.",
            "uses": "",
            "growing_tips": "",
            "ideal_conditions": "",
            "error": str(e)
        }

    response = jsonify(info)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)