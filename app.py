from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("sgd_svm_model.pkl")
scaler = joblib.load("scaler.pkl")


# Feature order must match training dataset columns
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

@app.route("/")
def index():
    return render_template("index.html")  # your HTML form

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    # Build feature vector for model
    row = []
    try:
        for feat in FEATURES:
            # MentHlth and PhysHlth are numeric but optional; provide default if missing
            if feat in ["MentHlth", "PhysHlth"]:
                val = float(data.get(feat, 0))
            else:
                val = float(data[feat])
            row.append(val)
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid input for {feat}"}), 400

    row = np.array(row).reshape(1, -1)

    row_scaled = scaler.transform(row)

    pred = model.predict(row_scaled)[0]

    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    app.run(debug=True)
