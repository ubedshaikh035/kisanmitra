from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("crop_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [[
        data["N"], data["P"], data["K"],
        data["temperature"], data["humidity"],
        data["ph"], data["rainfall"]
    ]]
    prediction = model.predict(features)[0]
    return jsonify({"crop": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)