from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)  # Convert input to array
    prediction = model.predict(features)[0]  # Get the prediction
    return jsonify({"predicted_demand": prediction})

if __name__ == "__main__":
    app.run(debug=True)
