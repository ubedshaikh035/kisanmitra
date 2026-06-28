from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

with open("crop_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)