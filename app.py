from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import json
import requests
import pandas as pd  

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load crop model
with open("crop_recommendation_model.pkl", "rb") as f:
    crop_model = pickle.load(f)

# Load fertilizer model and label encoder
with open("fertilizer_recommendation_model.pkl", "rb") as f:
    fertilizer_model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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
    prediction = crop_model.predict(features)[0]
    response = jsonify({"crop": prediction})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route("/fertilizer-predict", methods=["GET", "POST", "OPTIONS"])
def fertilizer_predict():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200
    data = request.get_json(force=True)
    import pandas as pd
    features = pd.DataFrame([{
        "Temperature": float(data["Temperature"]),
        "Moisture": float(data["Moisture"]),
        "Rainfall": float(data["Rainfall"]),
        "PH": float(data["PH"]),
        "Nitrogen": float(data["Nitrogen"]),
        "Phosphorous": float(data["Phosphorous"]),
        "Potassium": float(data["Potassium"]),
        "Carbon": float(data["Carbon"]),
        "Soil": data["Soil"],
        "Crop": data["Crop"]
    }])
    prediction_encoded = fertilizer_model.predict(features)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    response = jsonify({"fertilizer": prediction})
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
        text = result["choices"][0]["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        info = json.loads(text)
    except Exception as e:
        info = {"error": str(e)}

    response = jsonify(info)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
@app.route("/nearby-markets", methods=["GET", "POST", "OPTIONS"])
def nearby_markets():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    data = request.get_json(force=True)
    city = data.get("city", "")
    api_key = os.environ.get("GOOGLE_PLACES_API_KEY")

    try:
        # Geocode city
        geocode_res = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": city, "key": api_key}
        )
        geocode_data = geocode_res.json()

        if not geocode_data["results"]:
            response = jsonify({"error": "City not found"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

        lat = geocode_data["results"][0]["geometry"]["location"]["lat"]
        lng = geocode_data["results"][0]["geometry"]["location"]["lng"]

        # Find nearby markets
        places_res = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={
                "location": f"{lat},{lng}",
                "radius": 10000,
                "keyword": "farming market agriculture mandi",
                "key": api_key
            }
        )
        places_data = places_res.json()
        results = places_data.get("results", [])

        if not results:
            response = jsonify({"name": "No markets found nearby", "address": "Try a different city"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

        market = results[0]
        response = jsonify({
            "name": market["name"],
            "address": market.get("vicinity", "Address not available")
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
@app.route("/agri-news", methods=["GET", "POST", "OPTIONS"])

def agri_news():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response, 200

    # Accept query either as GET param or POST body, matching FlutterFlow flexibility
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        query = data.get("q", "agriculture")
    else:
        query = request.args.get("q", "agriculture")

    api_key = os.environ.get("GNEWS_API_KEY")

    try:
        gnews_res = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": query,
                "lang": "en",
                "max": 10,
                "apikey": api_key
            },
            timeout=10
        )
        gnews_data = gnews_res.json()
        response = jsonify(gnews_data)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)