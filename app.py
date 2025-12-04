from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load("tree_benefit_predictor_final.joblib")

benefit_names = [
	"Gross Carbon Sequestration (lb/yr)",
	"Avoided Runoff (gal/yr)",
	"Pollution Removal (oz/yr)",
	"Oxygen Production (lb/yr)",
	"Total Annual Benefits ($/yr)"
]

@app.route("/", methods=["GET"])
def home():
	return jsonify({"status": "Tree Benefit AI Model is running 🌳"})

@app.route("/predict", methods=["POST"])
def predict():
	try:
		data = request.get_json(force=True)
		df = pd.DataFrame([{
			"Species Name": data.get("species"),
			"DBH (in)": float(data.get("dbh")),
			"Height (ft)": float(data.get("height"))
		}])
		pred = model.predict(df)[0]
		results = dict(zip(benefit_names, [float(x) for x in pred]))
		return jsonify({
			"species": data.get("species"),
			"dbh": data.get("dbh"),
			"height": data.get("height"),
			"predicted_benefits": results
		})
	except Exception as e:
		return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)