from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Example: Load or train the model (if not already done)
from sklearn.ensemble import RandomForestRegressor
import pickle

# Sample model (Replace with your actual trained model)
model = RandomForestRegressor()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)  # Convert input to array
    prediction = model.predict(features)[0]  # Get the prediction
    return jsonify({"predicted_demand": prediction})

if __name__ == "__main__":
    app.run(debug=True)


# Train the model

# Assuming your trained model is stored in a variable named 'model'
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
