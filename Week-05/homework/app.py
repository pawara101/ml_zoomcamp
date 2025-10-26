from fastapi import FastAPI
from pydantic import BaseModel
import pickle

model_file_path = "pipeline_v1.bin"
with open(model_file_path, "rb") as model_file:
    dv, model = pickle.load(model_file)

app = FastAPI()

class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/get-prediction")
def get_prediction(customer_data: Customer):
    # Convert request body to dictionary
    customer_dict = customer_data.model_dump()

    # Transform and predict
    X = dv.transform([customer_dict])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    # Return result
    result = {'churn': bool(churn), 'probability': float(y_pred)}
    return result