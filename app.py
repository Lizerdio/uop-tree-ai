from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import JSONResponse

model = joblib.load("tree_benefit_predictor_final.joblib")

app = FastAPI(title="UOP Tree Benefit Predictor")

class TreeInput(BaseModel):
	species: str
	dbh: float
	height: float

@app.get("/")
def home():
	return {"message": "UOP Tree Benefit Predictor API is running!"}

@app.post("/predict")
def predict(tree: TreeInput):
	try:
		data = pd.DataFrame([{
			"Species Name": tree.species,
			"DBH (in)": tree.dbh,
			"Height": tree.height
		}])
		preds = model.predict(data)[0]
		return {
			"Gross Carbon Sequestration (lb/yr)": round(preds[0], 3),
			"Avoided Runoff (gal/yr)": round(preds[1], 3),
			"Pollution Removal (oz/yr)": round(preds[2], 3),
			"Oxygen Production (lb/yr)": round(preds[3], 3),
			"Total Annual Benefits ($/yr)": round(preds[4], 2)
		}
	except Exception as e:
		return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=5000)

