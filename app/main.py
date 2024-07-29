from fastapi import FastAPI, HTTPException, Request
from app.preprocess import PreProcessor
from app.model import ResumeCNNModel
import json

app = FastAPI()

# Pre-load models and other heavy initializations in the global scope if necessary
pre_processor = PreProcessor()
r_model = ResumeCNNModel()
model = r_model.build_model()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        text = body.get('text', '')

        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Preprocess the text
        matrix = pre_processor.text_to_matrix(text)

        # Reshape the matrix as expected by the model
        expected_shape = (1, pre_processor.cv_length, pre_processor.ncol)
        reshaped_m = matrix.reshape(expected_shape)

        # Make predictions
        pred_dic = {}
        for label in r_model.labels:
            modelx = model[label]
            predictions = modelx.predict(reshaped_m)
            pred_dic[label] = float(predictions[0][0])

        return pred_dic
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
