from fastapi import FastAPI
import uvicorn
import mlflow
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
mlflow.set_tracking_uri('http://ec2-54-165-160-184.compute-1.amazonaws.com:5000')

model_name = 'rent_model'
model_stage = 'Production'
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

class Data(BaseModel):
    BHK: int = 2
    Size: int = 1000
    Area_Type: str = 'Carpet Area'
    City: str = 'Kolkata'
    Furnishing_Status: str = 'Unfurnished'
    Tenant_Preferred: str = 'Bachelors/Family'
    Bathroom: int = 2
    Point_of_Contact: str = 'Contact Owner'


def create_dataframe(data):
    df = pd.DataFrame(data, index=[0])
    df = df.rename(columns={
        'BHK': 'BHK',
        'Size': 'Size',
        'Area_Type': 'Area Type',
        'City': 'City',
        'Furnishing_Status': 'Furnishing Status',
        'Tenant_Preferred': 'Tenant Preferred',
        'Bathroom': 'Bathroom',
        'Point_of_Contact': 'Point of Contact'
    })
    return df

@app.post("/predict")
def predict(data: Data):
    data = create_dataframe(dict(data))
    prediction = model.predict(data)
    return {"prediction": prediction[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

